"""Scripted speculative decode blocks for engine tests."""

from typing import Any

from ds4py.engine.decode import Verdict
from ds4py.errors import EngineError


def block_chunks(tokens: Any) -> Any:
    tokens = list(tokens)
    return [tokens[start + 1 : start + 17] for start in range(0, len(tokens), 17)]


class FakeMultiTokenStep:
    hooks_every_sample = True

    def __init__(self, blocks: Any = ()) -> None:
        self.blocks = iter(blocks)
        self.block_sizes = []

    def _drafts(self, sampled: Any) -> Any:
        provider: Any = next(self.blocks, ())
        if callable(provider):
            provider = provider(sampled)
        if provider is None:
            return None
        return tuple(provider)[:16]

    def run(self, session: Any, ctx: Any) -> None:
        while not ctx.cancelled():
            ctx.before_sample()
            policy = ctx.sampling()
            sampled = session.sample(
                policy.temperature,
                policy.top_k,
                policy.top_p,
                policy.min_p,
                ctx.rng,
            )
            if sampled < 0:
                raise EngineError("cannot sample: the session state is not valid; sync first")
            if ctx.is_stop(sampled):
                ctx.accept(sampled)
                return
            drafts = self._drafts(sampled)
            if drafts is None:
                return
            if ctx._job.logits_hooks:
                if self._verify_hooked(session, ctx, sampled, drafts):
                    return
                continue
            first_verdict = ctx.accept(sampled)
            tokens = (sampled, *drafts)
            block_start = session.pos()
            for token in tokens:
                session.eval(token)
            self.block_sizes.append(len(tokens))
            ctx.buffer_events()
            committed = 0
            for index, token in enumerate(tokens):
                if ctx.cancelled():
                    session.rewind(block_start + committed)
                    ctx.flush_events()
                    return
                verdict = first_verdict if index == 0 else ctx.accept(token)
                if verdict is Verdict.STOP_BEFORE:
                    session.rewind(block_start + committed)
                    ctx.flush_events()
                    return
                verdict = ctx.commit(token, verdict)
                committed += 1
                if verdict is Verdict.STOP_AFTER:
                    if session.pos() > block_start + committed:
                        session.rewind(block_start + committed)
                    ctx.flush_events()
                    return
            ctx.flush_events()

    def _verify_hooked(self, session: Any, ctx: Any, sampled: Any, drafts: Any) -> bool:
        # Verify each draft against a fresh sample with the current committed
        # parser/history and logits. A mismatch keeps the replacement token,
        # cuts this run, and returns to the next block without stale drafts.
        kept = 0
        for index, expected in enumerate((sampled, *drafts)):
            if ctx.cancelled():
                self.block_sizes.append(kept)
                return True
            token = sampled
            if index:
                ctx.before_sample()
                p = ctx.sampling()
                token = session.sample(p.temperature, p.top_k, p.top_p, p.min_p, ctx.rng)
                if token < 0:
                    raise EngineError("cannot sample: sync first")
            verdict = ctx.accept(token)
            if verdict is Verdict.STOP_BEFORE:
                self.block_sizes.append(kept)
                return True
            session.eval(token)
            verdict = ctx.commit(token, verdict)
            kept += 1
            if verdict is Verdict.STOP_AFTER:
                self.block_sizes.append(kept)
                return True
            if token != expected:
                break
        self.block_sizes.append(kept)
        return False
