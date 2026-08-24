from __future__ import annotations

import argparse

from datasets import Dataset
from transformers import AutoTokenizer

from cpo_trl.data import load_training_rows
from cpo_trl.finite import FiniteTrainingCallback
from cpo_trl.formatting import format_sft_text
from cpo_trl.peft import (
    load_causal_lm_for_training,
    lora_settings_from_config,
    peft_config_for_new_adapter,
)
from cpo_trl.trl_compat import ensure_trl_optional_dependency_stubs
from scripts.train.common import add_common_args, parse_with_config, training_args_dict

ensure_trl_optional_dependency_stubs()
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parse_with_config(parser)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_training_rows(args.train_file, "sft")
    dataset = Dataset.from_list([{"text": format_sft_text(tokenizer, row)} for row in rows])
    lora_settings = lora_settings_from_config(args)
    model = load_causal_lm_for_training(args.model_name_or_path, use_lora=args.use_lora)
    trainer_args = SFTConfig(
        **training_args_dict(args),
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config_for_new_adapter(
            args.model_name_or_path,
            use_lora=args.use_lora,
            lora_settings=lora_settings,
        ),
        callbacks=[FiniteTrainingCallback()],
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
