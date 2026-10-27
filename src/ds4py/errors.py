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


class ModelFileError(EngineError):
    """ModelFile failure."""


class EngineLockedError(EngineError):
    """EngineLocked failure."""


class CheckpointError(Ds4pyError):
    """Checkpoint failure."""


class StoreError(Ds4pyError):
    """Session store failure."""
