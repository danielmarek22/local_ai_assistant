import unittest
from unittest.mock import patch

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

    def test_multipurpose_commands_require_approval(self):
        tool = BashExecutionTool(timeout=5)

        commands = (
            "git reset --hard",
            "git clean -fd",
            "find . -delete",
            "sed -i s/old/new/ file.txt",
            "sort -o output.txt input.txt",
            "awk 'BEGIN { system(\"touch marker\") }'",
            "env sh -c 'touch marker'",
            "rg --pre 'sh -c \"touch marker\"' pattern .",
        )

        for command in commands:
            with self.subTest(command=command):
                needs_approval, reason = tool._approval_reason(command)
                self.assertTrue(needs_approval)
                self.assertIn("outside the instant read-only allowlist", reason)

    def test_embedded_newline_requires_approval(self):
        tool = BashExecutionTool(timeout=5)

        needs_approval, reason = tool._approval_reason("pwd\nprintf unsafe")

        self.assertTrue(needs_approval)
        self.assertIn("shell operator", reason)

    @patch("app.tools.bash_execution.subprocess.run")
    def test_instant_command_executes_without_a_shell(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "/tmp\n"
        run.return_value.stderr = ""
        tool = BashExecutionTool(timeout=5)

        result = tool.run("pwd")

        self.assertIn("Command succeeded", result)
        run.assert_called_once_with(
            ["pwd"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_empty_command_is_denied_without_approval(self):
        tool = BashExecutionTool(timeout=5)
        approvals = []

        result = tool.run("   ", approval_callback=lambda request: approvals.append(request) or True)

        self.assertEqual(approvals, [])
        self.assertIn("Permission Denied", result)


if __name__ == "__main__":
    unittest.main()
