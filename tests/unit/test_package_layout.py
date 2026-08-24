from __future__ import annotations


def test_new_internal_import_paths_are_available() -> None:
    from cpo_trl.data.datasets import load_jsonl
    from cpo_trl.data.formatting import format_prompt
    from cpo_trl.evaluation.teacher_forced import sequence_logps
    from cpo_trl.losses.cpo import cpo_unary_pair_loss
    from cpo_trl.losses.dpo_reference import dpo_pair_loss
    from cpo_trl.losses.kto_reference import kto_unary_loss
    from cpo_trl.metrics.preference import grouped_preference_record
    from cpo_trl.models.peft import LoraSettings
    from cpo_trl.references.ema import ClusterReferenceZ
    from cpo_trl.references.logratio import sequence_logratio
    from cpo_trl.references.token_kl import sampled_kl_regularizer
    from cpo_trl.sampling.pair_sampler import CPOPairAwareBatchSampler
    from cpo_trl.trainers.cpo_trainer import CPOTrainer
    from cpo_trl.utils.finite import assert_finite_tensor

    assert load_jsonl is not None
    assert format_prompt is not None
    assert sequence_logps is not None
    assert cpo_unary_pair_loss is not None
    assert dpo_pair_loss is not None
    assert kto_unary_loss is not None
    assert grouped_preference_record is not None
    assert LoraSettings is not None
    assert ClusterReferenceZ is not None
    assert sequence_logratio is not None
    assert sampled_kl_regularizer is not None
    assert CPOPairAwareBatchSampler is not None
    assert CPOTrainer is not None
    assert assert_finite_tensor is not None


def test_legacy_import_paths_remain_available() -> None:
    from cpo_trl.cpo_trainer import CPOTrainer
    from cpo_trl.data import load_jsonl
    from cpo_trl.eval import sequence_logps
    from cpo_trl.finite import assert_finite_tensor
    from cpo_trl.formatting import format_prompt
    from cpo_trl.losses import cpo_unary_pair_loss
    from cpo_trl.metrics import grouped_preference_record
    from cpo_trl.peft import LoraSettings
    from cpo_trl.sampling import CPOPairAwareBatchSampler
    from cpo_trl.trl_compat import ensure_trl_optional_dependency_stubs

    assert CPOTrainer is not None
    assert load_jsonl is not None
    assert sequence_logps is not None
    assert assert_finite_tensor is not None
    assert format_prompt is not None
    assert cpo_unary_pair_loss is not None
    assert grouped_preference_record is not None
    assert LoraSettings is not None
    assert CPOPairAwareBatchSampler is not None
    assert ensure_trl_optional_dependency_stubs is not None
