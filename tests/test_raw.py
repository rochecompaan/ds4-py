import io
import signal
import threading
from datetime import datetime

import pytest
import ds4py
from ds4py import cli, EngineError
from ds4py._native.fake import FakeNative, reply
from ds4py.engine.sampling import SamplingPolicy
from ds4py.raw import Trace, read_trace_tokens, run_raw

import ds4py.raw as raw_mod

def raw(prompt="hi", *, ctx=64, max_tokens=50, script=reply(b"ok"), trace=None, cancel=None):
    out, err = io.BytesIO(), io.StringIO()
    with ds4py.open("f.gguf", ctx=ctx, cancel=cancel, native=FakeNative(script=script)) as e:
        code = run_raw(e, prompt, max_tokens=max_tokens, sampling=e.sampling_defaults,
                       out=out, err=err, trace=trace)
    return code, out.getvalue(), err.getvalue()

def test_writes_the_token_bytes_and_nothing_else():
    assert raw(script=reply(b"hello")) == (0, b"hello", "")
    assert raw(script=reply(b"hello"), max_tokens=3) == (0, b"hel", "")

def test_a_prompt_that_does_not_fit_exits_2():
    assert raw("x" * 64, ctx=64) == (2, b"", "ds4py: prompt of 64 tokens does not fit a context of 64\n")

def test_an_empty_prompt_is_an_error():
    with pytest.raises(EngineError, match="raw prompt tokenized to no tokens"):
        raw("")

def test_ctrl_c_flushes_the_text_and_exits_130():
    main, cancel = threading.get_ident(), threading.Event()
    def script(prompt, generated):
        if len(generated) == 2:                              # Ctrl+C while the third token is sampled
            signal.pthread_kill(main, signal.SIGINT)
            assert cancel.wait(5)
        return ord("abc"[len(generated)]) + 1
    before = signal.getsignal(signal.SIGINT)
    assert raw(script=script, cancel=cancel) == (130, b"abc", "")
    assert signal.getsignal(signal.SIGINT) is before

def test_trace_uses_the_ds4_agent_format(tmp_path):
    path = tmp_path / "trace.log"
    for _ in range(2):                                       # the file is appended to; the reader takes the last run
        with Trace(path, now=lambda: datetime(2026, 10, 21, 9, 5, 7, 42_000)) as t:
            raw("a\n", script=reply(b'"\x7f\xc3'), trace=t)
    assert path.read_bytes().splitlines()[:7] == [
        b'2026-10-21 09:05:07.042 raw-prompt="a\\n"',
        b"2026-10-21 09:05:07.042 tokens label=raw_prompt_tokens start=0 len=2",
        b'2026-10-21 09:05:07.042 token index=0 id=98 bytes=1 text="a" hex=61',
        b'2026-10-21 09:05:07.042 token index=1 id=11 bytes=1 text="\\n" hex=0a',
        b'2026-10-21 09:05:07.042 token index=1 id=35 bytes=1 text="\\"" hex=22',
        b'2026-10-21 09:05:07.042 token index=2 id=128 bytes=1 text="\\x7f" hex=7f',
        b'2026-10-21 09:05:07.042 token index=3 id=196 bytes=1 text="\xc3" hex=c3',
    ]
    assert read_trace_tokens(path) == ([98, 11], [35, 128, 196])

@pytest.fixture
def opened(monkeypatch):
    seen = {}
    def open_engine(model, **kw):
        seen.update(model=model, **kw)
        return ds4py.open(model, ctx=kw["ctx"], native=FakeNative(script=reply(b"ok")))
    monkeypatch.setattr(raw_mod, "open_engine", open_engine)
    return seen

def test_cli_passes_the_flags_through(opened, monkeypatch):
    ran = []
    monkeypatch.setattr(raw_mod, "run_raw", lambda engine, prompt, **kw: ran.append((prompt, kw)) or 0)
    assert cli.main(["raw", "hi", "-m", "m.gguf", "-c", "64", "-n", "5", "--temp", "0",
                     "--top-p", "0.9", "--seed", "1", "--cpu", "-t", "2"]) == 0
    assert opened == {"model": "m.gguf", "ctx": 64, "backend": "cpu", "seed": 1, "threads": 2, "lib": None}
    (prompt, kw), = ran
    assert (prompt, kw["max_tokens"], kw["trace"]) == ("hi", 5, None)
    assert kw["sampling"] == SamplingPolicy(temperature=0.0, top_p=0.9, seed=1)

def test_cli_defaults_and_prompt_sources(opened, capsysbinary, tmp_path):
    (tmp_path / "p.txt").write_text("hi")
    assert cli.main(["raw", "--prompt-file", str(tmp_path / "p.txt")]) == 0
    assert capsysbinary.readouterr().out == b"ok"
    assert opened == {"model": "ds4flash.gguf", "ctx": 100_000, "backend": None, "seed": None, "threads": 0, "lib": None}
    with pytest.raises(SystemExit): cli.main(["raw", "hi", "-p", "hi"])     # exactly one prompt source
    with pytest.raises(SystemExit): cli.main(["raw"])


def test_trace_reader_accepts_c_style_lines_and_requires_raw_run(tmp_path):
    path = tmp_path / "c.log"
    path.write_bytes(b'2026-10-21 09:05:07.042 unrelated\n')
    with pytest.raises(ValueError, match="no raw run"):
        read_trace_tokens(path)
    with path.open("ab") as stream:
        stream.write(b'2026-10-21 09:05:07.042 tokens label=raw_prompt_tokens start=0 len=1\n'
                     b'2026-10-21 09:05:07.042 token index=0 id=-1 bytes=0 text="" hex=\n'
                     b'2026-10-21 09:05:07.042 token index=1 id=42 bytes=1 text="x" hex=78\n')
    assert read_trace_tokens(path) == ([-1], [42])


def test_raw_can_run_off_the_main_thread():
    result = []
    thread = threading.Thread(target=lambda: result.append(raw()))
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert result == [(0, b"ok", "")]


def test_cli_reports_errors_without_a_traceback(opened, capsysbinary, tmp_path):
    assert cli.main(["raw", "-p", ""]) == 1
    assert capsysbinary.readouterr().err == b"ds4py: raw prompt tokenized to no tokens\n"
    assert cli.main(["raw", "--prompt-file", str(tmp_path / "missing")]) == 1
    assert b"ds4py:" in capsysbinary.readouterr().err


def test_cli_trace_and_explicit_prompt(opened, capsysbinary, tmp_path):
    path = tmp_path / "trace.log"
    assert cli.main(["raw", "-p", "a", "--trace", str(path), "--min-p", "0"]) == 0
    assert capsysbinary.readouterr().out == b"ok"
    assert read_trace_tokens(path) == ([98], [112, 108])


def test_context_reserve_stops_successfully():
    code, text, err = raw(ctx=64, max_tokens=500, script=lambda p, g: 98)
    assert code == 0
    assert 0 < len(text) < 500
    assert text == b"a" * len(text)
    assert err == ""
