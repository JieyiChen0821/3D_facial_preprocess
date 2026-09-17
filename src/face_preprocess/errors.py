from __future__ import annotations


class FacePreprocessError(Exception):
    """Base class for classified pipeline errors."""


class UsageContractError(FacePreprocessError):
    """CLI or configuration contract error detected before processing."""


class ObjFormatError(FacePreprocessError):
    """Input OBJ cannot be represented by the strict triangular geometry contract."""


class ManifestError(FacePreprocessError):
    """Input manifest is invalid or ambiguous."""


class ConfigError(FacePreprocessError):
    """Configuration is invalid."""


class ArtifactError(FacePreprocessError):
    """A checkpoint, template, or other shared artifact is invalid."""


class SampleFailure(FacePreprocessError):
    """A single sample failed while the batch can continue."""

    def __init__(self, message: str, *, contract: str | None = None) -> None:
        super().__init__(message)
        self.contract = contract


class RunFatalError(FacePreprocessError):
    """A shared runtime failure requires the whole run to stop."""
