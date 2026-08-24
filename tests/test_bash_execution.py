import unittest

from app.tools.bash_execution import BashExecutionTool


class BashExecutionToolTests(unittest.TestCase):
    def test_read_only_allowlisted_command_runs_without_approval(self):
        tool = BashExecutionTool(timeout=5)
        approvals = []

        result = tool.run("pwd", approval_callback=lambda request: approvals.append(request) or False)

        self.assertEqual(approvals, [])
        self.assertIn("Command succeeded.", result)

    def test_unrecognized_command_requires_and_uses_approval(self):
        tool = BashExecutionTool(timeout=5)
        approvals = []

        def approve(request):
            approvals.append(request)
            return True

        result = tool.run("printf approved", approval_callback=approve)

        self.assertEqual(len(approvals), 1)
        self.assertEqual(approvals[0]["tool"], "execute_bash")
        self.assertIn("outside the instant read-only allowlist", approvals[0]["reason"])
        self.assertEqual(approvals[0]["command"], "printf approved")
        self.assertIn("approved", result)

    def test_unrecognized_command_is_denied_without_human_approval(self):
        tool = BashExecutionTool(timeout=5)

        result = tool.run("printf denied", approval_callback=lambda _request: False)

        self.assertIn("Permission Denied: Human approval was not granted.", result)

    def test_shell_operator_requires_approval(self):
        tool = BashExecutionTool(timeout=5)
        approvals = []

        result = tool.run("pwd && pwd", approval_callback=lambda request: approvals.append(request) or False)

        self.assertEqual(len(approvals), 1)
        self.assertIn("shell operator '&&'", approvals[0]["reason"])
        self.assertIn("Permission Denied", result)

    def test_empty_command_is_denied_without_approval(self):
        tool = BashExecutionTool(timeout=5)
        approvals = []

        result = tool.run("   ", approval_callback=lambda request: approvals.append(request) or True)

        self.assertEqual(approvals, [])
        self.assertIn("Permission Denied", result)


if __name__ == "__main__":
    unittest.main()
