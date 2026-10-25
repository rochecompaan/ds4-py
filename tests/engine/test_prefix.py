from pathlib import Path

import pytest

from ds4py.engine.prefix import PrefixError, PrefixTurn, load_prefix, parse_prefix


BOM = b"\xef\xbb\xbf"


def test_one_pair_and_multiline_content():
    assert parse_prefix(b"USER: first\nsecond\nASSISTANT: answer\nnext\n") == (
        PrefixTurn("user", "first\nsecond"),
        PrefixTurn("assistant", "answer\nnext"),
    )


def test_multiple_pairs_preserve_order():
    assert parse_prefix(b"USER:u1\nASSISTANT:a1\nUSER:u2\nASSISTANT:a2") == (
        PrefixTurn("user", "u1"),
        PrefixTurn("assistant", "a1"),
        PrefixTurn("user", "u2"),
        PrefixTurn("assistant", "a2"),
    )


def test_bom_is_recognized_only_at_the_start():
    data = "USER: héllo\nASSISTANT: ✓".encode()
    assert parse_prefix(BOM + data) == (PrefixTurn("user", "héllo"), PrefixTurn("assistant", "✓"))
    with pytest.raises(PrefixError, match="line 1 must start"):
        parse_prefix(BOM + BOM + data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "prefix file is empty"),
        (BOM, "prefix file contains no turns"),
        (b"USER: x\0\nASSISTANT: y", "NUL byte"),
    ],
)
def test_empty_nul_and_no_turn_inputs_are_errors(data, message):
    with pytest.raises(PrefixError, match=message):
        parse_prefix(data)


def test_turn_marker_must_start_at_column_zero():
    with pytest.raises(PrefixError, match="line 1 must start with USER: or ASSISTANT:"):
        parse_prefix(b" USER: x\nASSISTANT: y")


@pytest.mark.parametrize(
    ("data", "line", "role"),
    [
        (b"ASSISTANT: x\nUSER: y", 1, "USER"),
        (b"USER: x\nUSER: y", 2, "ASSISTANT"),
        (b"USER: x\nASSISTANT: y\nASSISTANT: z", 3, "USER"),
    ],
)
def test_roles_alternate_starting_with_user(data, line, role):
    with pytest.raises(PrefixError, match=rf"line {line}: expected {role}:"):
        parse_prefix(data)


def test_exactly_one_optional_space_or_tab_is_removed():
    assert parse_prefix(b"USER:  two\nASSISTANT:\t\tvalue") == (
        PrefixTurn("user", " two"),
        PrefixTurn("assistant", "\tvalue"),
    )


def test_marker_like_text_inside_a_line_is_content():
    assert parse_prefix(b"USER: say ASSISTANT: hi\nASSISTANT: USER: ok") == (
        PrefixTurn("user", "say ASSISTANT: hi"),
        PrefixTurn("assistant", "USER: ok"),
    )


def test_only_one_newline_and_one_preceding_cr_are_trimmed():
    assert parse_prefix(b"USER: one\r\nASSISTANT: two\r\n") == (
        PrefixTurn("user", "one"), PrefixTurn("assistant", "two"),
    )
    assert parse_prefix(b"USER: one\n\nASSISTANT: two\n\n") == (
        PrefixTurn("user", "one\n"), PrefixTurn("assistant", "two\n"),
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"USER:\nASSISTANT: x", "line 1 has an empty USER turn"),
        (b"USER: \nASSISTANT: x", "line 1 has an empty USER turn"),
        (b"USER: x\nASSISTANT:", "line 2 has an empty ASSISTANT turn"),
        (b"USER: x", "prefix file must end with an ASSISTANT turn"),
    ],
)
def test_empty_turns_and_a_final_user_turn_are_errors(data, message):
    with pytest.raises(PrefixError, match=message):
        parse_prefix(data)


def test_whitespace_only_turn_with_remaining_content_is_accepted():
    assert parse_prefix(b"USER:  \nASSISTANT: \t\n") == (
        PrefixTurn("user", " "), PrefixTurn("assistant", "\t"),
    )


def test_invalid_utf8_reports_the_source_line():
    with pytest.raises(PrefixError, match="line 3: invalid UTF-8 in ASSISTANT turn") as caught:
        parse_prefix(b"USER: ok\nASSISTANT: first\n\xff")
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_load_prefix_reads_bytes(tmp_path):
    path = tmp_path / "prefix.txt"
    path.write_bytes(BOM + "USER: héllo\nASSISTANT: ✓".encode())
    assert load_prefix(path) == (PrefixTurn("user", "héllo"), PrefixTurn("assistant", "✓"))


def test_load_open_failure_includes_path_and_cause(tmp_path):
    path = tmp_path / "missing.txt"
    with pytest.raises(PrefixError, match=rf"cannot open prefix file .*{path.name}") as caught:
        load_prefix(path)
    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_load_read_failure_includes_path_and_cause(tmp_path, monkeypatch):
    path = tmp_path / "broken.txt"

    class BrokenReader:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self):
            raise OSError("read failed")

    monkeypatch.setattr(Path, "open", lambda self, *args: BrokenReader())
    with pytest.raises(PrefixError, match=rf"cannot read prefix file .*{path.name}: read failed") as caught:
        load_prefix(path)
    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)


def test_load_parse_failure_includes_path(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"USER: x")
    with pytest.raises(PrefixError, match="must end with an ASSISTANT turn") as caught:
        load_prefix(path)
    assert str(path) in str(caught.value)
