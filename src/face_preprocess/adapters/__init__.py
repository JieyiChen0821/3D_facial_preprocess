"""Stable, explicitly registered model-adapter interface."""

from face_preprocess.adapters.registry import (
    AdapterSpec,
    ParameterSpec,
    adapter_source_sha256,
    build_adapter,
    get_adapter_spec,
    list_adapter_info,
    register_adapter,
    resolve_adapter_configuration,
)

__all__ = [
    "AdapterSpec",
    "ParameterSpec",
    "adapter_source_sha256",
    "build_adapter",
    "get_adapter_spec",
    "list_adapter_info",
    "register_adapter",
    "resolve_adapter_configuration",
]
