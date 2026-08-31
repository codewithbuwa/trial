from __future__ import annotations

import argparse

from datasets import Dataset
from transformers import AutoTokenizer

from cpo_trl.data.datasets import load_training_rows
from cpo_trl.utils.finite import FiniteTrainingCallback
from cpo_trl.data.formatting import format_kto_row
from cpo_trl.metrics.preference import GroupedPreferenceMetricsCallback, SparseTrainingPrinterCallback
from cpo_trl.models.peft import (
    load_causal_lm_for_training,
    lora_settings_from_config,
    peft_config_for_new_adapter,
)
from cpo_trl.utils.trl_compat import ensure_trl_optional_dependency_stubs
from cpo_trl.utils.run_manifest import write_run_manifest
from common import add_common_args, parse_with_config, training_args_dict

ensure_trl_optional_dependency_stubs()
from transformers import PrinterCallback, ProgressCallback
from trl import KTOConfig, KTOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--beta", type=float, default=argparse.SUPPRESS)
    args = parse_with_config(parser)
    if not hasattr(args, "beta"):
        args.beta = 0.1
    write_run_manifest(args.output_dir, method="kto", args=args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_training_rows(args.train_file, "kto")
    dataset = Dataset.from_list([format_kto_row(tokenizer, row) for row in rows])
    lora_settings = lora_settings_from_config(args)
    model = load_causal_lm_for_training(args.model_name_or_path, use_lora=args.use_lora)
    trainer_args = KTOConfig(
        **training_args_dict(args),
        beta=args.beta,
        max_length=args.max_seq_length,
    )
    trainer = KTOTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config_for_new_adapter(
            args.model_name_or_path,
            use_lora=args.use_lora,
            lora_settings=lora_settings,
        ),
        callbacks=[
            FiniteTrainingCallback(fail_fast=True),
            GroupedPreferenceMetricsCallback(method="kto", beta=args.beta),
        ],
    )
    if args.terminal_log_steps:
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(SparseTrainingPrinterCallback(every_n_steps=args.terminal_log_steps))
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
