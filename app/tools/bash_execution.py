import subprocess
import logging
import shlex
from typing import Callable, TypedDict

logger = logging.getLogger("bash_tool")


class BashApprovalRequest(TypedDict):
    tool: str
    command: str
    reason: str


class BashExecutionTool:
    """
    Executes bash commands on the local system.

    Common read-only commands run immediately. Everything else requires an
    explicit human approval callback before it is passed to bash.
    """
    
    name = "execute_bash"
    description = (
        "Executes local bash commands. Common read-only commands run immediately; "
        "commands with writes, shell operators, or unrecognized executables require browser approval first."
    )

    # Add the JSON schema parameters for Ollama
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute"
            }
        },
        "required": ["command"]
    }

    # Common read-only commands that can run without interrupting the user.
    INSTANT_READ_ONLY_COMMANDS = {
        "awk",
        "basename",
        "cat",
        "date",
        "df",
        "dirname",
        "du",
        "env",
        "find",
        "free",
        "git",
        "head",
        "id",
        "jq",
        "ls",
        "pwd",
        "rg",
        "sed",
        "sort",
        "stat",
        "tail",
        "uname",
        "uniq",
        "uptime",
        "wc",
        "whoami",
    }

    # Shell operators make a command harder to reason about, so they need approval.
    APPROVAL_OPERATORS = {";", "&&", "||", "|", ">", ">>", "<", "$(", "`"}

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return True

    def _approval_reason(self, command: str) -> tuple[bool, str]:
        """
        Return whether the command needs human approval and why.

        Empty or malformed commands are still denied outright because there is
        nothing useful for a human to approve.
        """
        for op in self.APPROVAL_OPERATORS:
            if op in command:
                return True, f"Command contains shell operator '{op}'."
        
        try:
            tokens = shlex.split(command)
            if not tokens:
                raise ValueError("Empty command.")
            
            base_cmd = tokens[0]
            if base_cmd not in self.INSTANT_READ_ONLY_COMMANDS:
                return True, f"Command '{base_cmd}' is outside the instant read-only allowlist."
            
            return False, ""
        except ValueError as e:
            raise ValueError(f"Malformed command: {e}") from e

    def run(
        self,
        command: str,
        approval_callback: Callable[[BashApprovalRequest], bool] | None = None,
    ) -> str | None:
        logger.info("Evaluating bash command: %s", command)
        
        try:
            needs_approval, reason = self._approval_reason(command)
        except ValueError as exc:
            logger.warning("Blocked malformed bash command: %s (Reason: %s)", command, exc)
            return f"Permission Denied: {exc}"

        if needs_approval:
            if approval_callback is None:
                logger.warning("Blocked bash command without approval handler: %s (Reason: %s)", command, reason)
                return f"Permission Denied: {reason}"

            approved = approval_callback({
                "tool": self.name,
                "command": command,
                "reason": reason,
            })
            if not approved:
                logger.info("Human denied bash command: %s", command)
                return f"Permission Denied: Human approval was not granted. Reason: {reason}"

            logger.info("Human approved bash command: %s", command)
        else:
            logger.info("Command classified as read-only. Executing without approval.")

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                executable="/bin/bash"
            )
            
            output = result.stdout.strip()
            errors = result.stderr.strip()
            
            if result.returncode == 0:
                return f"Command succeeded.\nOutput:\n{output}" if output else "Command succeeded with no output."
            else:
                return f"Command failed with return code {result.returncode}.\nError:\n{errors}\nOutput:\n{output}"
                
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout} seconds."
        except Exception as e:
            logger.error("Bash execution error: %s", e)
            return f"System error executing command: {str(e)}"
