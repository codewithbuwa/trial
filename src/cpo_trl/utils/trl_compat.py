"""Compatibility shims for optional TRL imports."""

from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any


def ensure_optional_mergekit_stub() -> None:
    """Provide a tiny mergekit stub for TRL versions that import it eagerly.

    TRL uses mergekit only for optional model-merging callbacks. Some releases
    import mergekit at trainer import time, which breaks DPO/KTO training when
    mergekit is not installed. The real mergekit package currently conflicts
    with this repo's pinned Accelerate version, so we expose the symbols TRL
    imports and fail only if the unused merge path is actually called.
    """

    if "mergekit" in sys.modules or importlib.util.find_spec("mergekit") is not None:
        return

    mergekit = types.ModuleType("mergekit")
    config = types.ModuleType("mergekit.config")
    merge = types.ModuleType("mergekit.merge")
    mergekit.__spec__ = ModuleSpec("mergekit", loader=None)
    config.__spec__ = ModuleSpec("mergekit.config", loader=None)
    merge.__spec__ = ModuleSpec("mergekit.merge", loader=None)

    class MergeConfiguration(dict):
        pass

    class MergeOptions:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    def run_merge(*args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "mergekit is not installed. Install a compatible mergekit environment "
            "only if you need TRL model-merging callbacks."
        )

    config.MergeConfiguration = MergeConfiguration
    merge.MergeOptions = MergeOptions
    merge.run_merge = run_merge
    mergekit.config = config
    mergekit.merge = merge
    sys.modules["mergekit"] = mergekit
    sys.modules["mergekit.config"] = config
    sys.modules["mergekit.merge"] = merge


def ensure_optional_llm_blender_stub() -> None:
    """Provide a tiny llm_blender stub for TRL versions that import judges eagerly."""

    if "llm_blender" in sys.modules or importlib.util.find_spec("llm_blender") is not None:
        return

    llm_blender = types.ModuleType("llm_blender")
    llm_blender.__spec__ = ModuleSpec("llm_blender", loader=None)

    class Blender:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "llm_blender is not installed. Install it only if you need "
                "TRL's llm-blender judge integration."
            )

    llm_blender.Blender = Blender
    sys.modules["llm_blender"] = llm_blender


def ensure_optional_weave_stub() -> None:
    """Provide a tiny weave stub for TRL callback imports."""

    if "weave" in sys.modules or importlib.util.find_spec("weave") is not None:
        return

    weave = types.ModuleType("weave")
    trace = types.ModuleType("weave.trace")
    context = types.ModuleType("weave.trace.context")
    weave.__spec__ = ModuleSpec("weave", loader=None)
    trace.__spec__ = ModuleSpec("weave.trace", loader=None)
    context.__spec__ = ModuleSpec("weave.trace.context", loader=None)

    class EvaluationLogger:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "weave is not installed. Install it only if you need TRL's "
                "weave callback integration."
            )

    class _WeaveClientContext:
        def get(self) -> None:
            return None

    context.weave_client_context = _WeaveClientContext()
    weave.EvaluationLogger = EvaluationLogger
    weave.trace = trace
    trace.context = context
    sys.modules["weave"] = weave
    sys.modules["weave.trace"] = trace
    sys.modules["weave.trace.context"] = context


def ensure_trl_optional_dependency_stubs() -> None:
    """Register optional dependency stubs needed before importing TRL trainers."""

    ensure_optional_mergekit_stub()
    ensure_optional_llm_blender_stub()
    ensure_optional_weave_stub()
