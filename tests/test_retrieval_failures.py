import tempfile
import unittest
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.core.context_builder import ContextBuilder
from app.core.events import AssistantSpeechEvent, AssistantTurnFailureEvent, UserMessageAcceptedEvent
from app.core.orchestrator import Orchestrator
from app.memory.chat_history import ChatHistoryStore
from app.memory.retriever import MemoryRetriever
from app.memory.summary_store import SummaryStore
from app.memory.memory_store import MemoryStore, MemoryIndexSyncError
from app.storage.database import Database
from app import server
from app.transport.connection_hub import SessionConnectionHub


class RetrievalFailureTests(unittest.TestCase):
    def test_stale_deleted_vectors_do_not_enter_the_generation_prompt(self):
        semantic = Mock()
        memory = MemoryStore(self.db, SimpleNamespace(semantic_collection=semantic))
        memory_id = memory.add("Deleted semantic fact")
        semantic.delete.side_effect = RuntimeError("Vector cleanup unavailable")
        with self.assertRaises(MemoryIndexSyncError):
            memory.delete_memories([memory_id])
        self.collection.delete.side_effect = RuntimeError("Vector cleanup unavailable")
        with self.assertLogs("chat_history", level="ERROR"):
            self.history.delete_session("past-session")
        semantic.query.return_value = {"ids": [[memory_id]], "distances": [[0.1]],
                                       "documents": [["Deleted semantic fact"]]}
        self.collection.query.return_value = {"ids": [[f"message:{self.past_message_id}"]],
                                             "distances": [[0.1]], "documents": [["Episodic fact"]]}
        self.retriever.memory = memory
        events = list(self.orchestrator.handle_user_input("session", "Question"))
        prompt = str(self.llm.stream_chat.call_args.args[0])
        self.assertNotIn("Deleted semantic fact", prompt)
        self.assertNotIn("Episodic fact", prompt)
        self.assertTrue(any(isinstance(event, AssistantSpeechEvent) and event.is_final for event in events))

    def test_refresh_before_acceptance_does_not_abort_persisted_turn(self):
        self.configure_queries(())

        async def check():
            hub = SessionConnectionHub()
            app = SimpleNamespace(state=SimpleNamespace(connection_hub=hub))
            dead = SimpleNamespace(app=app, send_text=AsyncMock(side_effect=RuntimeError("Disconnected")))
            hub.register("session", "origin", dead)
            with patch.object(server, "synthesize_async", AsyncMock(side_effect=RuntimeError("No TTS"))):
                await server._stream_orchestrator_events(
                    dead, self.orchestrator,
                    self.orchestrator.handle_user_input("session", "Question"), "origin", 0,
                )
            rows = self.history.get_all("session")
            self.assertEqual([row["role"] for row in rows], ["user", "assistant"])
            replacement = SimpleNamespace(send_text=AsyncMock())
            hub.register("session", "replacement", replacement)
            await hub.replay_pending("replacement")
            frames = [json.loads(call.args[0]) for call in replacement.send_text.await_args_list]
            self.assertEqual(frames[0], {"type": "user_message_accepted", "message_id": rows[0]["id"], "is_retry": False})
            self.assertEqual(frames[1]["type"], "assistant_end")
            self.assertEqual(frames[1]["content"], "A useful answer.")

        asyncio.run(check())

    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.close)
        uploads = tempfile.TemporaryDirectory()
        self.addCleanup(uploads.cleanup)
        self.collection = Mock()
        self.history = ChatHistoryStore(
            self.db, SimpleNamespace(episodic_collection=self.collection),
            uploads_root=uploads.name,
        )
        self.past_message_id = self.history.add("past-session", "user", "Episodic fact")
        self.memory = Mock()
        self.retriever = MemoryRetriever(self.memory, self.history)
        self.summary = SummaryStore(self.db)
        self.llm = Mock()
        self.llm.stream_chat = Mock(return_value=iter(["A useful answer."]))
        self.orchestrator = Orchestrator(
            self.llm, ContextBuilder("Answer the user.", self.history, summary_store=self.summary),
            self.history, self.summary,
            SimpleNamespace(collect_context=lambda **kwargs: None),
            self.retriever, SimpleNamespace(finalize=lambda *args, **kwargs: None),
        )

    def configure_queries(self, failures, *, check_persisted=False):
        def semantic(*args, **kwargs):
            if check_persisted:
                self.assertEqual(self.history.count_messages("session"), 1)
            if "semantic" in failures:
                raise RuntimeError("semantic unavailable")
            return ["Semantic fact"]

        def episodic(**kwargs):
            if check_persisted:
                self.assertEqual(self.history.count_messages("session"), 1)
            if "episodic" in failures:
                raise RuntimeError("episodic unavailable")
            return {"ids": [[f"message:{self.past_message_id}"]],
                    "documents": [["Episodic fact"]], "distances": [[0.1]]}

        self.memory.get_relevant.side_effect = semantic
        self.collection.query.side_effect = episodic

    def test_each_provider_fails_independently_and_reports_degradation(self):
        for failures in (("semantic",), ("episodic",), ("semantic", "episodic")):
            with self.subTest(failures=failures):
                self.configure_queries(failures)
                with self.assertLogs("memory_retriever", level="ERROR"):
                    result = self.retriever.retrieve("Question", "session")
                self.assertEqual(result.failed_sources, failures)
                for source in ("semantic", "episodic"):
                    self.assertEqual(
                        f"{source.title()} fact" in (result.memory_context or ""),
                        source not in failures,
                    )
                self.assertIn("unavailable", result.perception_value)

    def test_turn_saves_input_before_queries_and_answers_during_total_outage(self):
        self.configure_queries(("semantic", "episodic"), check_persisted=True)
        iterator = self.orchestrator.handle_user_input("session", "Question")
        accepted = next(event for event in iterator if isinstance(event, UserMessageAcceptedEvent))
        self.memory.get_relevant.assert_not_called()
        self.collection.query.assert_not_called()
        self.assertEqual(self.history.get_all("session")[0]["id"], accepted.message_id)
        with self.assertLogs("memory_retriever", level="ERROR"):
            events = list(iterator)
        self.assertFalse(any(isinstance(event, AssistantTurnFailureEvent) for event in events))
        self.assertTrue(any(
            isinstance(event, AssistantSpeechEvent) and event.is_final
            and event.text == "A useful answer." for event in events
        ))
        self.assertEqual([row["role"] for row in self.history.get_all("session")], ["user", "assistant"])
        self.assertEqual(self.llm.stream_chat.call_args.args[0][-1]["content"], "Question")

    def test_persistence_failure_does_not_query_memory_or_acknowledge_input(self):
        self.configure_queries(())
        self.history.add = Mock(side_effect=RuntimeError("storage unavailable"))
        events = []
        with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
            for event in self.orchestrator.handle_user_input("session", "Question"):
                events.append(event)
        self.assertFalse(any(isinstance(event, UserMessageAcceptedEvent) for event in events))
        self.memory.get_relevant.assert_not_called()
        self.collection.query.assert_not_called()
        self.llm.stream_chat.assert_not_called()

    def test_partial_outage_preserves_healthy_provider_and_saved_summary_in_prompt(self):
        for failures in (("semantic",), ("episodic",)):
            with self.subTest(failures=failures):
                session = failures[0]
                self.summary.set(session, "Earlier conversation summary", 0)
                self.configure_queries(failures)
                self.llm.stream_chat.return_value = iter(["A useful answer."])
                with self.assertLogs("memory_retriever", level="ERROR"):
                    events = list(self.orchestrator.handle_user_input(session, "Question"))
                prompt = self.llm.stream_chat.call_args.args[0]
                self.assertIn("Earlier conversation summary", prompt[0]["content"])
                for source in ("semantic", "episodic"):
                    self.assertEqual(
                        f"{source.title()} fact" in prompt[0]["content"], source not in failures,
                    )
                self.assertFalse(any(isinstance(event, AssistantTurnFailureEvent) for event in events))
                self.assertEqual(self.history.count_messages(session), 2)

    def test_retry_during_outage_reuses_saved_input(self):
        message_id = self.history.add("session", "user", "Question")
        self.history.mark_turn_failed("session", message_id, "Previous failure")
        self.configure_queries(("semantic", "episodic"), check_persisted=True)
        with self.assertLogs("memory_retriever", level="ERROR"):
            events = list(self.orchestrator.handle_user_input(
                "session", "Question", existing_user_message_id=message_id,
            ))
        accepted = next(event for event in events if isinstance(event, UserMessageAcceptedEvent))
        self.assertTrue(accepted.is_retry)
        self.assertEqual(accepted.message_id, message_id)
        self.assertEqual([row["role"] for row in self.history.get_all("session")], ["user", "assistant"])
        self.assertIsNone(self.history.get_retryable_user_message("session", message_id))

    def test_recovered_providers_are_queried_again_without_stale_failure_state(self):
        self.configure_queries(("semantic", "episodic"))
        with self.assertLogs("memory_retriever", level="ERROR"):
            self.retriever.retrieve("Question", "session")
        self.configure_queries(())
        result = self.retriever.retrieve("Question", "session")
        self.assertEqual(result.failed_sources, ())
        self.assertIn("Semantic fact", result.memory_context)
        self.assertIn("Episodic fact", result.memory_context)
        self.assertNotIn("unavailable", result.perception_value)


if __name__ == "__main__":
    unittest.main()
