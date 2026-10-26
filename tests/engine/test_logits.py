import threading
from array import array
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.events import Stopped, Token, ToolCalls, Thinking
from ds4py.engine.logits import LogitsView
from ds4py.errors import EngineError


def scripted(*tokens):
    return lambda prompt, generated: (
        tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS
    )


@contextmanager
def fake_session(script, *, chat=False, think=ThinkMode.HIGH):
    with ds4py.open(
        Path("f.gguf"),
        ctx=1024,
        think=think,
        native=FakeNative(script=script),
    ) as engine:
        session = engine.session()
        if chat:
            session.start_chat()
        else:
            session.append_tokens([5, 6], "user")
        yield session


def native_session(session):
    return session._worker.call(lambda engine, native: native).result()


def emitted_ids(events):
    return [event.id for event in events if isinstance(event, Token)]


def test_no_hooks_never_copy_logits():
    with fake_session(scripted(7, FAKE_EOS)) as session:
        native = native_session(session)
        assert list(session.run_round())[-1] == Stopped("stop_token")
        assert native.logit_copies == 0


def test_sampled_tokens_materialize_only_on_access():
    class CountingSequence(Sequence):
        def __init__(self, values):
            self.values = values
            self.iterations = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

        def __iter__(self):
            self.iterations += 1
            return iter(self.values)

    tokens = CountingSequence([7, 8, 9])
    view = LogitsView(object(), vocab_size=1)
    view._update(
        position=3, in_think=False, think_tokens=0, parser_state=None,
        sampled_tokens=tokens,
    )
    assert tokens.iterations == 0

    snapshot = view.sampled_tokens
    assert snapshot == (7, 8, 9)
    assert view.sampled_tokens is snapshot
    assert tokens.iterations == 1


def test_logits_view_reuses_one_buffer_and_validates_lengths():
    seen = {}

    def hook(view):
        first, second = view.logits(), view.logits()
        seen["buffer"] = (first.obj, second.obj)
        seen["size"] = len(first)
        view.set_logits(first)
        with pytest.raises(ValueError, match="vocabulary size"):
            view.set_logits(array("f", [0.0]))
        with pytest.raises(ValueError, match="vocabulary size"):
            view.set_logits(array("f", [0.0]) * (len(first) + 1))
        for token in (-1, len(first)):
            with pytest.raises(ValueError, match="token ID"):
                view.bias(token, 1.0)
        with pytest.raises(TypeError, match="token ID"):
            view.bias(True, 1.0)

    with fake_session(scripted(FAKE_EOS)) as session:
        session.add_logits_hook(hook)
        assert list(session.run_round())[-1] == Stopped("stop_token")
    assert seen["buffer"][0] is seen["buffer"][1]
    assert isinstance(seen["buffer"][0], array)
    assert seen["size"] == 279


def test_bias_changes_the_actual_next_token_and_transcript():
    mutations = []

    def hook(view):
        if not view.sampled_tokens:
            before = view.logits()[7]
            view.bias(8, 2.0)
            mutations.append((before, view.logits()[8]))

    with fake_session(scripted(7, FAKE_EOS)) as session:
        session.add_logits_hook(hook)
        events = list(session.run_round())
        assert emitted_ids(events) == [8]
        assert session.transcript.tokens == (5, 6, 8)
        assert mutations == [(1.0, 2.0)]
        assert native_session(session).logit_sets == 1


def test_hook_state_tracks_only_committed_tokens():
    tokens = tokenize_rendered("<think>a</think>b<think>c</think>")

    with fake_session(
        scripted(*tokens),
        chat=True,
        think=ThinkMode.HIGH,
    ) as session:
        prompt_length = len(session.transcript.tokens)
        # run_round appends the assistant prefix before the first hook.
        assistant_prefix = session._worker.call(
            lambda engine, native: engine.assistant_prefix(
                session._transcript.tokens, ThinkMode.NONE
            )
        ).result()
        initial_position = prompt_length + len(assistant_prefix)
        states = []

        def hook(view):
            states.append(
                (
                    view.position,
                    view.in_think,
                    view.think_tokens,
                    view.parser_state,
                    view.sampled_tokens,
                )
            )

        session.add_logits_hook(hook)
        events = list(session.run_round())
        assert emitted_ids(events) == tokens
        assert [state[0] for state in states] == [
            initial_position + index for index in range(len(tokens) + 1)
        ]
        assert [state[1] for state in states] == [
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
        ]
        assert [state[2] for state in states] == [0, 1, 2, 2, 2, 1, 2, 2]
        assert [state[3] for state in states] == ["SEARCH"] * (len(tokens) + 1)
        assert [state[4] for state in states] == [
            tuple(tokens[:index]) for index in range(len(tokens) + 1)
        ]


def test_hooks_share_a_view_and_run_in_order_on_the_worker_thread():
    calls, views = [], []

    def first(view):
        calls.append(("first", threading.get_ident()))
        views.append(view)

    def second(view):
        calls.append(("second", threading.get_ident()))
        views.append(view)

    with fake_session(scripted(7, FAKE_EOS)) as session:
        worker_thread = session.on_worker(
            lambda handles: threading.get_ident()
        ).result()
        session.add_logits_hook(first)
        session.add_logits_hook(second)
        list(session.run_round())
        assert [name for name, thread in calls] == ["first", "second"] * 2
        assert {thread for name, thread in calls} == {worker_thread}
        assert all(view is views[0] for view in views)
        assert native_session(session).logit_copies == 0


def test_hook_error_commits_nothing_and_releases_round_ownership():
    error = RuntimeError("hook failed")
    calls = 0

    def hook(view):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error

    with fake_session(scripted(7, 8, FAKE_EOS)) as session:
        session.add_logits_hook(hook)
        with pytest.raises(RuntimeError) as caught:
            list(session.run_round())
        assert caught.value is error
        assert session.transcript.tokens == (5, 6)
        events = list(session.run_round())
        assert emitted_ids(events) == [7, 8]
        assert events[-1] == Stopped("stop_token")


def test_installing_a_hook_during_an_active_round_is_refused():
    with fake_session(scripted(7, 8, FAKE_EOS)) as session:
        events = session.run_round()
        assert next(event for event in events if isinstance(event, Token)).id == 7
        with pytest.raises(EngineError, match="active round"):
            session.add_logits_hook(lambda view: None)
        events.close()
        session.add_logits_hook(lambda view: None)


def test_literal_think_in_tool_argument_does_not_change_hook_control_state():
    text = '<tool_call>read<arg_key>path</arg_key><arg_value><think>x</arg_value></tool_call>'
    tokens = [byte + 1 for byte in text.encode()]
    states = []
    with ds4py.open('f.gguf', think=ThinkMode.NONE,
                    native=FakeNative(script=scripted(*tokens), info=replace(
                        FakeNative()._info, family='glm', is_glm_dsa=True))) as engine:
        session = engine.session()
        session.append_user('read')
        session.add_logits_hook(lambda view: states.append((view.in_think, view.think_tokens)))
        events = list(session.run_round())
    assert events[-1] == Stopped('tool_call_complete')
    assert len(states) == len(tokens) + 1
    assert all(state == (False, 0) for state in states)
    assert not any(isinstance(event, Thinking) for event in events)
    calls = next(event.calls for event in events if isinstance(event, ToolCalls))
    assert calls[0].args == {'path': '<think>x'} and calls[0].raw == text
