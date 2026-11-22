import json

import pytest

from ds4py.engine.tool_syntax.hermes import Hermes, repair_hermes_quotes

WRITE_BODY = '{"name": "write", "arguments": {"path": "a.py", "content": "print("hi")\nx = ",".join(r)\n"}}'
WRITE_REPAIRED = '{"name": "write", "arguments": {"path": "a.py", "content": "print(\\"hi\\")\nx = \\",\\".join(r)\n"}}'
EDIT_BODY = '{"name": "edit", "arguments": {"path": "r.py", "edits": [{"oldText": "",".join(str(v) for v in r)", "newText": "", ".join(str(v) for v in r)"}, {"oldText": "say "a"", "newText": "say \'a\'"}]}}'
EDIT_REPAIRED = '{"name": "edit", "arguments": {"path": "r.py", "edits": [{"oldText": "\\",\\".join(str(v) for v in r)", "newText": "\\", \\".join(str(v) for v in r)"}, {"oldText": "say \\"a\\"", "newText": "say \'a\'"}]}}'
LEGACY_BODY = '{"name": "edit", "arguments": {"path": "r.py", "oldText": "",".join(r)", "newText": "", ".join(r)"}}'
LEGACY_REPAIRED = (
    '{"name": "edit", "arguments": {"path": "r.py", "oldText": "\\",\\".join(r)", "newText": "\\", \\".join(r)"}}'
)


@pytest.mark.parametrize(
    ("body", "repaired"),
    [(WRITE_BODY, WRITE_REPAIRED), (EDIT_BODY, EDIT_REPAIRED), (LEGACY_BODY, LEGACY_REPAIRED)],
    ids=["write", "edit-list", "edit-legacy"],
)
def test_reference_quote_forms_repair_exactly(body: str, repaired: str) -> None:
    assert repair_hermes_quotes(body) == repaired


@pytest.mark.parametrize("chunk_size", [1, 7, 4096], ids=lambda size: f"chunk-{size}")
@pytest.mark.parametrize(
    ("body", "repaired"),
    [(WRITE_BODY, WRITE_REPAIRED), (EDIT_BODY, EDIT_REPAIRED), (LEGACY_BODY, LEGACY_REPAIRED)],
    ids=["write", "edit-list", "edit-legacy"],
)
def test_repaired_calls_parse_with_original_raw(body: str, repaired: str, chunk_size: int) -> None:
    data = ("<tool_call>\n" + body + "\n</tool_call>").encode()
    parser = Hermes()
    for offset in range(0, len(data), chunk_size):
        parser.feed(data[offset : offset + chunk_size])
    snapshot = parser.finish()
    assert snapshot.state == "DONE", snapshot.error
    assert len(snapshot.calls) == 1
    call = snapshot.calls[0]
    assert dict(call.args) == json.loads(repaired, strict=False)["arguments"]
    assert call.raw == data.decode()


def test_valid_call_does_not_need_repair() -> None:
    body = '{"name": "write", "arguments": {"path": "a.py", "content": "print(\\"hi\\")"}}'
    assert repair_hermes_quotes(body) is None


REFUSALS = [
    '{"name": "write", "arguments": {"path": "a"b.py", "content": "x"}}',
    '{"name": "write", "arguments": {"path": "a.py", "content": "x", "path": "b.py", "content": "say "y""}}',
    '{"name": "edit", "arguments": {"path": "r.py", "edits": [{"oldText": "say "a"", "newText": "b"}, {"oldText": "z"}]}}',
    '{"name": "edit", "arguments": {"path": "r.py", "edits": [{"oldText": "say "a", "path": "x", "newText": "b"}]}}',
    '{"name": "edit", "arguments": {"path": "r.py", "oldText": "say "a", "path": "x", "newText": "b"}}',
    '{"name": "edit", "arguments": {"path": "r.py", "oldText": "a""}, {"oldText": "c", "newText": "b"}}',
    '{"name": "write", "arguments": {"path": "a.md", "content": `x`}}',
    '{"name": "write", "arguments": {"path": "a.py", "content": "a\x00b"}}',
    WRITE_BODY[:-2],
]


@pytest.mark.parametrize("body", REFUSALS)
def test_ambiguous_or_unsafe_forms_are_not_repaired(body: str) -> None:
    assert repair_hermes_quotes(body) is None


@pytest.mark.parametrize("body", REFUSALS)
def test_refused_forms_never_publish_a_call(body: str) -> None:
    parser = Hermes()
    parser.feed(("<tool_call>" + body + "</tool_call>").encode())
    snapshot = parser.finish()
    assert snapshot.state == "ERROR"
    assert snapshot.calls == ()


def test_truncated_repair_candidate_never_publishes() -> None:
    parser = Hermes()
    parser.feed(("<tool_call>" + WRITE_BODY).encode())
    snapshot = parser.finish()
    assert snapshot.state == "ERROR"
    assert snapshot.calls == ()


def test_backtick_error_names_its_argument() -> None:
    body = '{"name":"edit","arguments":{"path":"t.html","edits":[{"oldText":`<html>`,"newText":"x"}]}}'
    parser = Hermes()
    parser.feed(("<tool_call>" + body + "</tool_call>").encode())
    snapshot = parser.finish()
    assert snapshot.state == "ERROR"
    assert snapshot.error is not None
    assert "oldText" in snapshot.error and "backticks" in snapshot.error
