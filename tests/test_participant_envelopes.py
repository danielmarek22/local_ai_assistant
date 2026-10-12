import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.conversation import (
    AssistantEnvelopeError, AssistantEnvelopeFilter, GROUP_CONTEXT_INSTRUCTION, InputSource,
    SenderAttribution, SenderType, SessionKind, render_group_message,
    normalize_generated_assistant_output, unwrap_assistant_envelope,
)
from app.core.context_builder import ContextBuilder
from app.core.events import AssistantSpeechEvent
from app.core.response_generator import ResponseGenerator
from app.core.turn_finalizer import TurnFinalizer


SENDER = SenderAttribution('astra', 'Astra', SenderType.LOCAL_ASSISTANT,
                           InputSource.ASSISTANT_GENERATION)


def consume(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as end:
            return ''.join(e.text for e in events if isinstance(e, AssistantSpeechEvent)), end.value


class ParticipantEnvelopeTests(unittest.TestCase):
    def test_nested_envelopes_decode_json_escapes(self):
        text = 'Hello!\n\nUnicode: café. "Quoted."'
        wrapped = render_group_message(render_group_message(text, SENDER), SENDER)
        self.assertEqual(unwrap_assistant_envelope(wrapped, sender_id='astra'), text)

    def test_non_envelopes_and_other_senders_are_preserved(self):
        wrapped = render_group_message('Hello', SENDER)
        for text in (f'Example: {wrapped}', f'```json\n{wrapped}\n```',
                     wrapped + ' trailing text', wrapped[:-1],
                     wrapped.replace('local_assistant', 'human'),
                     'PARTICIPANT_MESSAGE {"content":"Hello"}'):
            with self.subTest(text=text):
                self.assertEqual(unwrap_assistant_envelope(text), text)
        self.assertEqual(unwrap_assistant_envelope(wrapped, sender_id='other'), wrapped)

    def test_stream_filter_handles_every_split_without_leaking_prefix(self):
        wrapped = render_group_message('Hello', SENDER)
        for index in range(len(wrapped) + 1):
            filt = AssistantEnvelopeFilter(True)
            self.assertEqual(filt.push(wrapped[:index]), '')
            self.assertEqual(filt.push(wrapped[index:]), '')
            self.assertEqual(filt.flush(), 'Hello')
        filt = AssistantEnvelopeFilter(True)
        self.assertEqual(filt.push('Hello'), 'Hello')
        self.assertEqual(filt.push(' there'), ' there')

    def test_streaming_and_buffered_paths_emit_and_return_clean_reply(self):
        wrapped = render_group_message('Hello there!', SENDER)
        llm = Mock(spec=["chat_buffered", "stream_chat"])
        llm.stream_chat.return_value = iter(wrapped)
        llm.chat_buffered.return_value = {'content': wrapped}
        generator = ResponseGenerator(llm, SimpleNamespace(get_native_tools=lambda **kwargs: []), Mock())
        messages = [{'role': 'system', 'content': GROUP_CONTEXT_INSTRUCTION}]
        speech, result = consume(generator.stream_response('group', messages))
        self.assertEqual((speech, result), ('Hello there!', 'Hello there!'))
        speech, result = consume(generator._stream_late_routing_step('group', messages, 'Hi'))
        self.assertEqual(speech, 'Hello there!')
        self.assertEqual(result.response, 'Hello there!')

    def test_invalid_generated_wrappers_raise_before_speech_in_both_paths(self):
        wrapped = render_group_message('Hello there!', SENDER)
        cases = (
            wrapped[:-2],
            wrapped.replace('Hello there!', 'Hello" \\n\\n"Second paragraph'),
            wrapped.replace('Hello there!', 'Bad\\q escape'),
            wrapped + ' trailing text',
            wrapped.replace('local_assistant', 'external_agent'),
            'PARTICIPANT_MESSAGE',
            'PARTICIPANT_MESSAGE{broken}',
            'PARTICIPANT_MESSAGE {"content":"Hello"}',
            '[state:happy] [animation:head_nod_yes] ' + wrapped[:-2],
        )
        messages = [{'role': 'system', 'content': GROUP_CONTEXT_INSTRUCTION}]
        for text in cases:
            for buffered in (False, True):
                with self.subTest(text=text, buffered=buffered):
                    llm = Mock(spec=['chat_buffered', 'stream_chat'])
                    llm.stream_chat.return_value = iter(text)
                    llm.chat_buffered.return_value = {'content': text}
                    generator = ResponseGenerator(
                        llm, SimpleNamespace(get_native_tools=lambda **kwargs: []), Mock(),
                    )
                    response = (generator._stream_late_routing_step('group', messages, 'Hi')
                                if buffered else generator.stream_response('group', messages))
                    events = []
                    with self.assertRaises(AssistantEnvelopeError):
                        for event in response:
                            events.append(event)
                    self.assertFalse(any(isinstance(e, AssistantSpeechEvent) for e in events))

    def test_avatar_prefixed_valid_wrapper_is_unwrapped_at_every_stream_split(self):
        wrapped = '[state:happy] ' + render_group_message('Hello!', SENDER)
        expected = '[state:happy] Hello!'
        self.assertEqual(normalize_generated_assistant_output(wrapped), expected)
        for index in range(len(wrapped) + 1):
            with self.subTest(index=index):
                filt = AssistantEnvelopeFilter(True)
                self.assertEqual(filt.push(wrapped[:index]), '')
                self.assertEqual(filt.push(wrapped[index:]), '')
                self.assertEqual(filt.flush(), expected)

    def test_output_policy_preserves_quoted_examples_and_normal_streaming(self):
        wrapped = render_group_message('Hello!', SENDER)
        for text in (f'Example: {wrapped}', f'```json\n{wrapped}\n```',
                     f'"{wrapped}"', '[state:happy] Hello!'):
            with self.subTest(text=text):
                self.assertEqual(normalize_generated_assistant_output(text), text)
                filt = AssistantEnvelopeFilter(True)
                self.assertEqual(''.join(filt.push(c) for c in text) + filt.flush(), text)
        filt = AssistantEnvelopeFilter(True)
        self.assertEqual(filt.push('[state:happy] Hello'), '[state:happy] Hello')
        self.assertEqual(filt.push(' there'), ' there')

    def test_direct_response_is_not_unwrapped(self):
        wrapped = render_group_message('Hello', SENDER)
        llm = Mock(spec=["chat_buffered", "stream_chat"])
        llm.chat_buffered.return_value = {'content': wrapped}
        generator = ResponseGenerator(llm, SimpleNamespace(get_native_tools=lambda **kwargs: []), Mock())
        _, result = consume(generator._stream_late_routing_step('direct', [], 'Hi'))
        self.assertIn('PARTICIPANT_MESSAGE', result.response)

    def test_context_and_summary_normalize_saved_assistant_only(self):
        wrapped = render_group_message(render_group_message('Hello', SENDER), SENDER)
        rows = [{'role': role, 'content': wrapped} for role in ('assistant', 'user')]
        history = SimpleNamespace(get_recent=lambda **kwargs: rows,
                                  effective_sender=lambda row: SENDER,
                                  get_session_kind=lambda session: SessionKind.MANUAL_GROUP)
        builder = ContextBuilder('System', history)
        messages = builder.build('group', '', session_kind=SessionKind.MANUAL_GROUP)
        contents = [json.loads(m['content'].split(' ', 1)[1])['content']
                    for m in messages if m['role'] != 'system']
        self.assertEqual(contents, ['Hello', wrapped])
        finalizer = TurnFinalizer(history, Mock(), Mock())
        self.assertEqual(finalizer._summary_content('group', rows[0]),
                         render_group_message('Hello', SENDER))
        self.assertEqual(finalizer._summary_content('group', rows[1]),
                         render_group_message(wrapped, SENDER))
