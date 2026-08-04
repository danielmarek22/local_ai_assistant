import logging

logger = logging.getLogger("tool_controls")


def build_prompt_with_tools(
    system_prompt: str,
    tool_executor,
) -> str:
    """
    Augment the system prompt with information about available tools.
    Similar to build_prompt_with_avatar_controls, this helps the assistant
    understand its capabilities outside of the ReAct loop.
    
    Args:
        system_prompt: The base system prompt
        tool_executor: ToolExecutor instance containing available tools
    
    Returns:
        Augmented system prompt with tool capability information
    """
    sections: list[str] = [system_prompt]
    
    # Collect available tools
    available_tools = []
    if tool_executor and hasattr(tool_executor, 'tools'):
        for name, tool in sorted(tool_executor.tools.items()):
            if getattr(tool, "is_available", False):
                available_tools.append(name)
    
    if available_tools:
        tool_names = ", ".join(available_tools)
        tool_block = (
            "## Available Tools\n"
            "You have access to the following tools to help answer user questions:\n"
            f"- {', '.join(available_tools)}\n\n"
            "Use these tools when you need current information or when explicitly requested. "
            "You can reference these capabilities in your responses and the user will understand that you can access them."
        )
        sections.append(tool_block)
    
    return "\n\n".join(section for section in sections if section)
