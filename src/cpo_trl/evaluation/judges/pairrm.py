from __future__ import annotations

from typing import Any


def _install_transformers_cache_shim() -> None:
    """Make old llm-blender importable under transformers>=5.

    llm-blender does ``from transformers.utils.hub import TRANSFORMERS_CACHE``,
    but that name was removed in transformers 5. Alias it to the current Hugging
    Face hub cache path so the import succeeds without downgrading transformers.
    """
    import transformers.utils.hub as hub

    if hasattr(hub, "TRANSFORMERS_CACHE"):
        return
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as cache_path
    except Exception:
        import os

        cache_path = os.environ.get(
            "HF_HUB_CACHE",
            os.path.expanduser("~/.cache/huggingface/hub"),
        )
    hub.TRANSFORMERS_CACHE = cache_path


def load_pairrm_ranker(model_name: str) -> Any:
    _install_transformers_cache_shim()
    try:
        import llm_blender
    except ImportError as exc:
        raise RuntimeError(
            "PairRM judge requires llm-blender. Install it with: "
            "poetry run pip install git+https://github.com/yuchenlin/LLM-Blender.git"
        ) from exc
    blender = llm_blender.Blender()
    blender.loadranker(model_name)
    return blender


def pairrm_judge(
    *,
    ranker: Any,
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
) -> dict[str, object]:
    prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
    ranks = ranker.rank(
        [prompt],
        [[response_a, response_b]],
        return_scores=False,
        batch_size=1,
    )
    first_rank, second_rank = [float(value) for value in ranks[0]]
    if first_rank < second_rank:
        winner, reason = "A", "PairRM ranked response A higher"
    elif second_rank < first_rank:
        winner, reason = "B", "PairRM ranked response B higher"
    else:
        winner, reason = "tie", "PairRM returned equal ranks"
    return {
        "winner": winner,
        "reason": reason,
        "status": "ok",
        "rank_a": first_rank,
        "rank_b": second_rank,
    }
