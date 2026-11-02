import ds4py

from ds4py._native.fake import FakeNative
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.tool_prompt import ToolDefinition


OPENING = (
    "You are a coding agent running in a local workspace. Use tools for local file and system work. "
    "Avoid printing large file contents or large code blocks as answers; create or edit files with tools, "
    "then summarize results briefly."
)
READ_GUIDANCE = (
    "read_file returns a bounded chunk, rather than unconditionally reading the whole file. "
    "For first looks at large files, prefer read_file with an explicit limit around 80-160. "
    "By default, read_file returns at most 160 lines and 50,000 characters. "
    "If the result says more lines are available, call read_file again with the offset shown in the result. "
    "A character-limited read ends on the last complete line that fits."
)


def test_default_prompt_aligns_agent_native_and_read_guidance() -> None:
    from ds4py.agent_prompt import DEFAULT_CODING_AGENT_INSTRUCTIONS

    definition = ToolDefinition(
        "read_file",
        "Read a file.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative()) as engine:
        segments = PromptBuilder(engine).begin(
            [definition],
            instructions=DEFAULT_CODING_AGENT_INSTRUCTIONS,
        )
        tokens = [token for segment in segments for token in segment.tokens]
        prompt = b"".join(engine.token_text(token) for token in tokens).decode()

    assert OPENING in prompt
    assert prompt.index(OPENING) < prompt.index("You have access to native DSML tools.")
    assert prompt.index("You have access to native DSML tools.") < prompt.index('{"name":"read_file"')
    assert READ_GUIDANCE in prompt

    assert "edit_file" in prompt
    assert "old_text" in prompt
    assert "new_text" in prompt
    assert "start_command" in prompt
    assert "check_command" in prompt
    assert "stop_command" in prompt
    assert "# Rules" in prompt
    assert "search_files" in prompt

    for forbidden in (
        "whole=true",
        "raw=true",
        "call more",
        "google_search",
        "visit_page",
    ):
        assert forbidden not in prompt
