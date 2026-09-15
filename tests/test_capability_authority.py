"""Model tool selection never grants authority to execute a capability."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.autonomy.store import AutonomyStore
from app.core.response_generator import ResponseGenerator
from app.core.tool_executor import ToolExecutor
from app.integrations import (
    CapabilityId, EventId, IntegrationEvent, IntegrationRegistry,
    InvocationContext, RegisteredTool, ToolCall, ToolResult, ToolSpec,
)


def consume(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


class CapabilityAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.capability = CapabilityId('demo', 'run')
        self.other = CapabilityId('demo', 'other')
        self.handler = Mock(return_value=ToolResult.success('done'))
        self.available = True
        tool = RegisteredTool(
            ToolSpec(self.capability, 'Harmless test tool', {
                'type': 'object', 'properties': {}, 'additionalProperties': False,
            }), self.handler, available=lambda: self.available,
        )
        self.registry = IntegrationRegistry([
            SimpleNamespace(name='demo', registered_tools=lambda: [tool]),
        ])
        self.call = ToolCall(self.capability, {})

    def test_direct_dispatch_rejects_empty_and_nonmatching_authority(self):
        for allowed in (frozenset(), frozenset({self.other})):
            with self.subTest(allowed=allowed):
                result = self.registry.invoke(self.call, InvocationContext(
                    'session', 'test', allowed_capabilities=allowed,
                ))
                self.assertEqual(result.status.value, 'denied')
        self.handler.assert_not_called()

    def test_allowed_and_unrestricted_participant_dispatch_still_work(self):
        for allowed in (None, frozenset({self.capability})):
            with self.subTest(allowed=allowed):
                result = self.registry.invoke(self.call, InvocationContext(
                    'session', 'test', allowed_capabilities=allowed,
                ))
                self.assertEqual(result.status.value, 'success')
        self.assertEqual(self.handler.call_count, 2)

    def test_context_copies_mutable_authority(self):
        original = set()
        context = InvocationContext('session', 'test', allowed_capabilities=original)
        original.add(self.capability)
        self.assertEqual(context.allowed_capabilities, frozenset())
        self.assertEqual(self.registry.invoke(self.call, context).status.value, 'denied')
        self.handler.assert_not_called()

    def test_event_dispatch_without_authority_fails_closed(self):
        result = self.registry.invoke(self.call, InvocationContext(
            'session', 'test', event_id='event',
        ))
        self.assertEqual(result.status.value, 'denied')
        self.handler.assert_not_called()

    def test_availability_is_rechecked_after_schema_discovery(self):
        self.assertEqual(len(self.registry.get_native_tools({self.capability})), 1)
        self.available = False
        result = self.registry.invoke(self.call, InvocationContext(
            'session', 'test', allowed_capabilities=frozenset({self.capability}),
        ))
        self.assertEqual(result.status.value, 'unavailable')
        self.handler.assert_not_called()

    def test_executor_journals_denial_without_running_handler(self):
        store = AutonomyStore(':memory:')
        self.addCleanup(store.close)
        executor = ToolExecutor(self.registry, operation_store=store)
        result = consume(executor.execute(
            self.call, 'session', 'test', allowed_capabilities=frozenset(),
        ))
        self.assertEqual(result.status.value, 'denied')
        self.handler.assert_not_called()
        self.assertEqual(store.pending_operations(), [])
        row = store._conn.execute('SELECT status FROM integration_operations').fetchone()
        self.assertEqual(row['status'], 'denied')

    def test_native_event_loop_enforces_authority_despite_model_call(self):
        event = IntegrationEvent(EventId('demo', 'wake'), {}, session_id='session')
        for allowed, expected in ((set(), 'denied'), ({self.other}, 'denied'),
                                  ({self.capability}, 'success'), (None, 'denied')):
            with self.subTest(allowed=allowed):
                self.handler.reset_mock()
                llm = SimpleNamespace(chat_buffered=Mock(side_effect=[
                    {'tool_calls': [{'function': {'name': str(self.capability), 'arguments': {}}}]},
                    {'content': 'Finished'},
                ]))
                messages = []
                generator = ResponseGenerator(llm, ToolExecutor(self.registry), Mock())
                answer = consume(generator.stream_late_routed_response(
                    'session', messages, 'test', allowed_capabilities=allowed,
                    event=event, persist_tool_traces=False,
                ))
                self.assertEqual(answer, 'Finished')
                observation = next(m for m in messages if m['role'] == 'tool')
                self.assertTrue(observation['content'].startswith(f'[{expected}]'))
                self.assertEqual(self.handler.call_count, int(expected == 'success'))

    def test_model_callback_cannot_expand_authority_mid_turn(self):
        allowed = set()
        def model(**kwargs):
            allowed.add(self.capability)
            return {'tool_calls': [{'function': {'name': str(self.capability), 'arguments': {}}}]}
        # Use a callable side effect so mutation occurs during the first inference.
        calls = 0
        def infer(**kwargs):
            nonlocal calls
            calls += 1
            return model(**kwargs) if calls == 1 else {'content': 'Finished'}
        llm = SimpleNamespace(chat_buffered=Mock(side_effect=infer))
        consume(ResponseGenerator(llm, ToolExecutor(self.registry), Mock()).stream_late_routed_response(
            'session', [], 'test', allowed_capabilities=allowed, persist_tool_traces=False,
        ))
        self.handler.assert_not_called()
