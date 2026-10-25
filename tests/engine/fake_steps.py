"""Scripted speculative decode blocks for engine tests."""
from ds4py.engine.decode import Verdict
from ds4py.errors import EngineError


def block_chunks(tokens):
    tokens = list(tokens)
    return [tokens[start + 1:start + 17] for start in range(0, len(tokens), 17)]


class FakeMultiTokenStep:
    hooks_every_sample = False

    def __init__(self, blocks=()):
        self.blocks = iter(blocks)
        self.block_sizes = []

    def _drafts(self, sampled):
        provider = next(self.blocks, ())
        if callable(provider):
            provider = provider(sampled)
        if provider is None:
            return None
        return tuple(provider)[:16]

    def run(self, session, ctx):
        while not ctx.cancelled():
            ctx.before_sample()
            policy = ctx.sampling()
            sampled = session.sample(
                policy.temperature, policy.top_k, policy.top_p, policy.min_p, ctx.rng,
            )
            if sampled < 0:
                raise EngineError("cannot sample: the session state is not valid; sync first")
            if ctx.is_stop(sampled):
                ctx.accept(sampled)
                return
            drafts = self._drafts(sampled)
            if drafts is None:
                return
            first_verdict = ctx.accept(sampled)
            tokens = (sampled, *drafts)
            block_start = session.pos()
            for token in tokens:
                session.eval(token)
            self.block_sizes.append(len(tokens))
            ctx.buffer_events()
            committed = 0
            for index, token in enumerate(tokens):
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
