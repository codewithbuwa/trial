from __future__ import annotations

import sys

from cpo_trl.trl_compat import ensure_trl_optional_dependency_stubs


def test_ensure_trl_optional_dependency_stubs_register_import_symbols() -> None:
    original = {
        name: sys.modules.get(name)
        for name in (
            "mergekit",
            "mergekit.config",
            "mergekit.merge",
            "llm_blender",
            "weave",
            "weave.trace",
            "weave.trace.context",
        )
    }
    for name in original:
        sys.modules.pop(name, None)

    try:
        ensure_trl_optional_dependency_stubs()
        from mergekit.config import MergeConfiguration
        from mergekit.merge import MergeOptions, run_merge
        import llm_blender
        from weave import EvaluationLogger
        from weave.trace.context import weave_client_context

        assert MergeConfiguration({"models": []}) == {"models": []}
        assert isinstance(MergeOptions(), MergeOptions)
        try:
            run_merge()
        except ImportError as exc:
            assert "mergekit is not installed" in str(exc)
        else:
            raise AssertionError("run_merge should fail without real mergekit")
        try:
            llm_blender.Blender()
        except ImportError as exc:
            assert "llm_blender is not installed" in str(exc)
        else:
            raise AssertionError("Blender should fail without real llm_blender")
        assert weave_client_context.get() is None
        try:
            EvaluationLogger()
        except ImportError as exc:
            assert "weave is not installed" in str(exc)
        else:
            raise AssertionError("EvaluationLogger should fail without real weave")
    finally:
        for name, module in original.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
