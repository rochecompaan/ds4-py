"""Typed errors raised by ds4py."""


class Ds4pyError(Exception):
    """Ds4py failure."""


class AbiError(Ds4pyError):
    """Abi failure."""


class TypeMapError(AbiError):
    """TypeMap failure."""


class AbiMismatch(AbiError):
    """AbiMismatch failure."""


class LibraryNotFound(AbiError):
    """LibraryNotFound failure."""


class LibraryAlreadyLoaded(AbiError):
    """LibraryAlreadyLoaded failure."""


class MissingBinding(AbiError):
    """MissingBinding failure."""


class BuildError(Ds4pyError):
    """Build failure."""


class EngineError(Ds4pyError):
    """Engine failure."""


class PayloadLoadError(EngineError):
    """A native payload could not be loaded as a verified prefix."""


class PayloadMismatch(EngineError):
    """Loaded native payload tokens differ from the expected tokens."""


class ContextExhausted(EngineError):
    """An open response reached the context reserve and no owner could make room."""


class ContextSelectionError(EngineError):
    """Rolling context selection failed with a stable diagnostic code."""

    def __init__(
        self,
        code: str,
        diagnostic: str,
        resident_tokens: int | None = None,
        estimated_tokens: int | None = None,
        budget_tokens: int | None = None,
    ) -> None:
        self.code = code
        self.diagnostic = diagnostic
        self.resident_tokens = resident_tokens
        self.estimated_tokens = estimated_tokens
        self.budget_tokens = budget_tokens
        super().__init__(f"[{code}] {diagnostic}")


class WorkerPoisonedError(EngineError):
    """The worker cannot run native operations after a fatal decode failure."""


class EmptyPromptError(EngineError):
    """The native session contract rejects an empty prompt."""


class ModelFileError(EngineError):
    """ModelFile failure."""


class EngineLockedError(EngineError):
    """EngineLocked failure."""


class CheckpointError(Ds4pyError):
    """Checkpoint failure."""


class StoreError(Ds4pyError):
    """Session store failure."""
