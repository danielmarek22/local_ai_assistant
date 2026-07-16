import unittest

from app.core.thinking_filter import ThinkingDirectiveFilter


class ThinkingDirectiveFilterTests(unittest.TestCase):
    def test_strips_tool_call_and_returns_directive(self):
        filter_ = ThinkingDirectiveFilter()

        visible, directives = filter_.push(
            'Need fresh facts. <tool_call>{"tool":"web_search","kwargs":{"query":"python news"}}</tool_call>'
        )
        final_visible, final_directives = filter_.flush()

        self.assertEqual(visible + final_visible, "Need fresh facts. ")
        self.assertEqual(len(directives + final_directives), 1)
        directive = (directives + final_directives)[0]
        self.assertEqual(directive.kind, "tool_call")
        self.assertEqual(directive.payload["tool"], "web_search")
        self.assertEqual(directive.payload["kwargs"]["query"], "python news")

    def test_handles_tool_call_split_across_chunks(self):
        filter_ = ThinkingDirectiveFilter()

        visible_a, directives_a = filter_.push("Need fresh facts. <tool_call>{\"tool\":\"web_search\",\"kwargs\":{\"query\":\"python news\"}}")
        visible_b, directives_b = filter_.push("</tool_call> done")
        visible_c, directives_c = filter_.flush()

        self.assertEqual(visible_a + visible_b + visible_c, "Need fresh facts.  done")
        self.assertEqual(directives_a, [])
        directives = directives_b + directives_c
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0].kind, "tool_call")
        self.assertEqual(directives[0].payload["tool"], "web_search")
        self.assertEqual(directives[0].payload["kwargs"]["query"], "python news")

    def test_handles_closing_tag_split_across_chunks(self):
        filter_ = ThinkingDirectiveFilter()

        filter_.push('<tool_call>{"tool":"web_search","kwargs":{"query":"weather"}}</tool')
        visible, directives = filter_.push("_call> ok")

        self.assertEqual(visible, " ok")
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0].payload["kwargs"]["query"], "weather")


if __name__ == "__main__":
    unittest.main()
