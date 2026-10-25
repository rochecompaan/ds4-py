"""Session-level chat helpers; these functions own transcript mutations."""
from ds4py.errors import EngineError
from .prompt import PromptBuilder


def _append(session, tokens, kind):
    if not tokens:
        return None
    return session._transcript.append(tokens, kind)


def start_chat(session, tools=(), instructions=None) -> None:
    if session._transcript.chat:
        raise EngineError("chat is already initialized")
    segments = PromptBuilder(session._engine).begin(tools, instructions)
    for segment in segments:
        _append(session, segment.tokens, segment.kind)
    session._transcript.set_chat()


def _start_if_empty(session) -> None:
    if session._transcript.chat:
        return
    if len(session._transcript):
        raise EngineError("chat is not initialized")
    start_chat(session)


def _require_chat(session) -> None:
    if not session._transcript.chat:
        raise EngineError("chat is not initialized")


def append_user(session, text, images=()):
    if images:
        raise EngineError("image input requires M5 vision support")
    _start_if_empty(session)
    builder = PromptBuilder(session._engine)
    return _append(session, builder.message(session._transcript.tokens, "user", text), "user")


def append_prefix(session, turns) -> None:
    turns = tuple(turns)
    if any(turn.role not in ("user", "assistant") for turn in turns):
        raise ValueError("prefix turns must have user or assistant roles")
    _start_if_empty(session)
    builder = PromptBuilder(session._engine)
    for turn in turns:
        tokens = builder.message(session._transcript.tokens, turn.role, turn.content)
        _append(session, tokens, turn.role)
        if turn.role == "assistant":
            append_assistant_end(session)


def append_assistant_prefix(session):
    _require_chat(session)
    builder = PromptBuilder(session._engine)
    tokens = builder.assistant_prefix(session._transcript.tokens)
    return _append(session, tokens, "assistant_prefix")


def append_assistant_end(session):
    _require_chat(session)
    return _append(session, PromptBuilder(session._engine).append_assistant_end(), "eos")


def append_tool_results(session, results):
    _require_chat(session)
    builder = PromptBuilder(session._engine)
    tokens = builder.tool_results(session._transcript.tokens, results)
    return _append(session, tokens, "tool")


def append_system_note(session, text):
    _require_chat(session)
    builder = PromptBuilder(session._engine)
    tokens = builder.message(session._transcript.tokens, "system", "\n" + text)
    return _append(session, tokens, "note")
