import subprocess
import logging
import shlex

logger = logging.getLogger("bash_tool")

class BashExecutionTool:
    """
    Executes bash commands on the local system with a strict allowlist.
    """
    
    name = "execute_bash"
    description = "Executes a safe read-only bash command. Allowed commands: ls, cat, pwd, whoami, uptime, df, free, date, head, tail."

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

    # Define strictly allowed base commands (read-only / safe)
    ALLOWED_COMMANDS = {
        "ls", "cat", "pwd", "whoami", "echo", "uptime", "df", "free", "date", "head", "tail"
    }

    # Block shell operators that allow command chaining/redirection/execution
    FORBIDDEN_OPERATORS = {";", "&&", "||", "|", ">", ">>", "<", "$(", "`"}

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return True

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Validates the command against the allowlist and checks for shell injection."""
        # 1. Check for forbidden operators to prevent chaining
        for op in self.FORBIDDEN_OPERATORS:
            if op in command:
                return False, f"Command contains forbidden shell operator '{op}'."
        
        # 2. Parse the command safely
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False, "Empty command."
            
            base_cmd = tokens[0]
            if base_cmd not in self.ALLOWED_COMMANDS:
                return False, f"Command '{base_cmd}' is not in the allowlist. Allowed: {', '.join(self.ALLOWED_COMMANDS)}."
            
            return True, ""
        except ValueError as e:
            # shlex raises ValueError for unclosed quotes, etc.
            return False, f"Malformed command: {e}"

    def run(self, command: str) -> str | None:
        logger.info("Evaluating bash command: %s", command)
        
        # Security Gate
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            logger.warning("Blocked unsafe bash command: %s (Reason: %s)", command, reason)
            return f"Permission Denied: {reason}"

        logger.info("Command passed security checks. Executing.")
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