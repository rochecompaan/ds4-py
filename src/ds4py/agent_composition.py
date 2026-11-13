"""Explicit composition contracts for the agent command and runtime."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

from ds4py.engine.sampling import SamplingPolicy

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace
    from collections.abc import Callable
    from pathlib import Path

    from ds4py.engine.session import Session
    from ds4py.runtime import AgentRuntime
    from ds4py.tui_commands import SlashCommandRegistry


@dataclass(frozen=True)
class EngineOpenOptions:
    """Generic engine, model, and sampling inputs resolved by the core CLI."""

    model: str
    ctx: int
    backend: str | None
    think: int
    seed: int | None
    threads: int
    lib: Path | None
    tokens: int
    interactive: bool
    temperature: float | None
    top_k: int | None
    top_p: float | None
    min_p: float | None


@dataclass(frozen=True)
class ConfiguredSession:
    """One configured session and the wrapper that owns its native resources."""

    session: Session
    sampling: SamplingPolicy
    close: Callable[[], None]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AgentExtension(Protocol):
    """Per-invocation hooks for composing a downstream agent command."""

    program: str

    def configure_parser(self, parser: ArgumentParser) -> None: ...

    def resolve_extension_config(self, args: Namespace, project_dir: Path) -> object: ...

    def open_session(
        self,
        options: EngineOpenOptions,
        project_dir: Path,
        extension_config: object,
    ) -> AbstractContextManager[ConfiguredSession]: ...

    def configure_commands(
        self,
        registry: SlashCommandRegistry,
        runtime: AgentRuntime,
    ) -> None: ...

    def before_identity_change(self) -> None: ...

    def after_identity_change(self) -> None: ...


class DefaultAgentExtension:
    """The unchanged core engine open path and no-op extension hooks."""

    program = "ds4py"

    def configure_parser(self, parser: ArgumentParser) -> None:
        pass

    def resolve_extension_config(self, args: Namespace, project_dir: Path) -> object:
        return None

    def open_session(
        self,
        options: EngineOpenOptions,
        project_dir: Path,
        extension_config: object,
    ) -> AbstractContextManager[ConfiguredSession]:
        from ds4py import open as open_engine  # noqa: PLC0415 - Preserve the CLI's monkeypatchable engine boundary.

        with ExitStack() as stack:
            engine = stack.enter_context(
                open_engine(
                    options.model,
                    ctx=options.ctx,
                    backend=options.backend,
                    think=options.think,
                    seed=options.seed,
                    threads=options.threads,
                    lib=options.lib,
                )
            )
            session = engine.session()
            sampling = SamplingPolicy.for_engine(
                engine.info,
                temperature=options.temperature,
                top_k=options.top_k,
                top_p=options.top_p,
                min_p=options.min_p,
                seed=options.seed,
            )
            return ConfiguredSession(session=session, sampling=sampling, close=stack.pop_all().close)

    def configure_commands(
        self,
        registry: SlashCommandRegistry,
        runtime: AgentRuntime,
    ) -> None:
        pass

    def before_identity_change(self) -> None:
        pass

    def after_identity_change(self) -> None:
        pass
