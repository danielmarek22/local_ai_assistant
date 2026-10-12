import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.context_builder import ContextBuilder
from app.core.conversation import (
    GROUP_CONTEXT_INSTRUCTION, InputSource, SenderAttribution, SenderType,
    SessionKind, relay_sender, render_group_message,
)
from app.core.events import AssistantSpeechEvent, AssistantTurnFailureEvent, UserMessageAcceptedEvent
from app.core.orchestrator import Orchestrator
from app.llm.base import InferenceFailure
from app.memory.chat_history import ChatHistoryStore
from app.memory.retriever import MemoryRetriever
from app.memory.summary_store import SummaryStore
from app.storage.database import Database


class RelayOutputRetryTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(':memory:')
        self.addCleanup(self.db.close)
        uploads = tempfile.TemporaryDirectory()
        self.addCleanup(uploads.cleanup)
        collection = Mock()
        collection.query.return_value = {'ids': [[]], 'documents': [[]], 'distances': [[]]}
        self.history = ChatHistoryStore(
            self.db, SimpleNamespace(episodic_collection=collection), uploads_root=uploads.name,
        )
        memory = Mock()
        memory.get_relevant.return_value = []
        summary = SummaryStore(self.db)
        self.llm = Mock(spec=['resolve_think_value', 'chat_buffered', 'stream_chat'])
        self.llm.resolve_think_value.return_value = True
        self.finalizer = Mock()
        self.orchestrator = Orchestrator(
            self.llm, ContextBuilder('Answer naturally.', self.history, summary_store=summary),
            self.history, summary,
            SimpleNamespace(collect_context=lambda **kwargs: None,
                            get_native_tools=lambda **kwargs: []),
            MemoryRetriever(memory, self.history), self.finalizer,
        )
        self.sender = relay_sender('external_agent', 'Peer Agent')
        self.valid = render_group_message(
            'A useful answer.',
            SenderAttribution('astra', 'Astra', SenderType.LOCAL_ASSISTANT,
                              InputSource.ASSISTANT_GENERATION),
        )

    def test_wrapper_failure_uses_existing_retry_and_reuses_relay_input(self):
        for buffered in (False, True):
            with self.subTest(buffered=buffered):
                session = f'relay-{buffered}'
                self.orchestrator.late_routing_enabled = buffered
                self.llm.reset_mock()
                self.finalizer.reset_mock()
                self.llm.chat_buffered.return_value = {'content': self.valid[:-2]}
                self.llm.stream_chat.return_value = iter(self.valid[:-2])
                with self.assertLogs('orchestrator', level='ERROR'):
                    events = list(self.orchestrator.handle_user_input(
                        session, 'Continue the story.', sender=self.sender,
                        session_kind=SessionKind.MANUAL_GROUP,
                    ))
                failure = next(e for e in events if isinstance(e, AssistantTurnFailureEvent))
                self.assertFalse(any(isinstance(e, AssistantSpeechEvent) for e in events))
                self.assertEqual([r['role'] for r in self.history.get_all(session)], ['user'])
                self.assertEqual(self.llm.chat_buffered.call_count, int(buffered))
                self.assertEqual(self.llm.stream_chat.call_count, int(not buffered))
                self.finalizer.finalize.assert_not_called()
                row = self.history.get_retryable_user_message(session, failure.user_message_id)
                self.assertEqual(row['content'], 'Continue the story.')
                self.assertEqual(row['sender_id'], self.sender.sender_id)
                self.assertEqual(row['retry_attempts'], 1)

                # A second malformed attempt remains retryable without adding history.
                self.llm.stream_chat.return_value = iter(self.valid[:-2])
                with self.assertLogs('orchestrator', level='ERROR'):
                    events = list(self.orchestrator.handle_user_input(
                        session, row['content'], sender=self.history.effective_sender(row),
                        session_kind=SessionKind.MANUAL_GROUP,
                        existing_user_message_id=row['id'],
                    ))
                failure = next(e for e in events if isinstance(e, AssistantTurnFailureEvent))
                self.assertEqual(failure.attempts, 2)
                self.assertEqual(self.history.count_messages(session), 1)

                # Retry only after the user's action; decode a valid generated wrapper.
                self.llm.chat_buffered.return_value = {'content': self.valid}
                self.llm.stream_chat.return_value = iter(self.valid)
                events = list(self.orchestrator.handle_user_input(
                    session, row['content'], sender=self.history.effective_sender(row),
                    session_kind=SessionKind.MANUAL_GROUP,
                    existing_user_message_id=row['id'],
                ))
                accepted = next(e for e in events if isinstance(e, UserMessageAcceptedEvent))
                self.assertTrue(accepted.is_retry)
                self.assertEqual(accepted.message_id, row['id'])
                self.assertIsNone(self.history.get_retryable_user_message(session, row['id']))
                rows = self.history.get_all(session)
                self.assertEqual([r['role'] for r in rows], ['user', 'assistant'])
                self.assertEqual(rows[1]['content'], 'A useful answer.')
                self.assertTrue(any(isinstance(e, AssistantSpeechEvent) and e.is_final
                                    and e.text == 'A useful answer.' for e in events))

    def test_group_recovery_wrapper_failure_is_retryable_without_leaking(self):
        self.orchestrator.late_routing_enabled = True
        self.llm.chat_buffered.side_effect = InferenceFailure('generation_deadline', 'Timed out')
        self.llm.stream_chat.return_value = iter(self.valid[:-2])
        with self.assertLogs('orchestrator', level='WARNING'):
            events = list(self.orchestrator.handle_user_input(
                'relay-recovery', 'Continue the story.', sender=self.sender,
                session_kind=SessionKind.MANUAL_GROUP,
            ))
        self.assertTrue(any(isinstance(e, AssistantTurnFailureEvent) for e in events))
        self.assertFalse(any(isinstance(e, AssistantSpeechEvent) for e in events))
        self.assertEqual([r['role'] for r in self.history.get_all('relay-recovery')], ['user'])
        self.assertEqual(self.llm.chat_buffered.call_count, 1)
        self.assertEqual(self.llm.stream_chat.call_count, 1)
        self.assertIn(GROUP_CONTEXT_INSTRUCTION, self.llm.stream_chat.call_args.args[0][0]['content'])


if __name__ == '__main__':
    unittest.main()
