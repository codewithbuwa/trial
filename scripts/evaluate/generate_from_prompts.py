from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from cpo_trl.data.formatting import apply_chat_template, chat_messages
from cpo_trl.models.peft import load_causal_lm_for_training


PROMPT_HEADER_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")


def parse_prompt_file(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL manifest or a text file containing numbered prompts."""

    if path.suffix == ".jsonl":
        prompts: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                instruction = str(row.get("instruction") or row.get("prompt") or "")
                if not instruction:
                    raise ValueError(f"{path}:{line_no}: missing instruction")
                prompts.append(
                    {
                        "prompt_id": str(row.get("prompt_id", len(prompts))),
                        "instruction": instruction,
                        "input": str(row.get("input", "")),
                        "cluster_id": str(row.get("cluster_id", "unknown")),
                    }
                )
        if not prompts:
            raise ValueError(f"no prompts found in {path}")
        return prompts

    prompts = []
    current_index: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_index, current_lines
        if current_index is None:
            return
        prompt = "\n".join(current_lines).strip()
        if prompt:
            prompts.append({"prompt_id": str(current_index), "instruction": prompt, "input": ""})
        current_index = None
        current_lines = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = PROMPT_HEADER_RE.match(raw_line)
        if match:
            flush()
            current_index = int(match.group(1))
            current_lines = [match.group(2)]
            continue
        if current_index is not None:
            current_lines.append(raw_line)
        elif raw_line.strip():
            raise ValueError(
                f"found text before first numbered prompt in {path}: {raw_line!r}"
            )
    flush()
    if not prompts:
        raise ValueError(f"no numbered prompts found in {path}")
    return prompts


def parse_model_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(value).name or value
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"invalid model spec: {value!r}")
        if name in seen:
            raise ValueError(f"duplicate model name: {name}")
        seen.add(name)
        specs.append((name, path))
    return specs


def validate_model_specs(specs: list[tuple[str, str]]) -> None:
    """Fail early when a local checkpoint path is missing."""

    missing: list[str] = []
    for name, model_path in specs:
        path = Path(model_path)
        looks_local = model_path.startswith((
            ".",
            "/",
            "output/",
            "outputs/",
            "output_compare/",
        ))
        if looks_local and not path.exists():
            missing.append(f"{name}={model_path}")
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"missing local checkpoint path(s):\n  {joined}")


def render_prompt(tokenizer: Any, instruction: str, input_text: str = "") -> str:
    return apply_chat_template(
        tokenizer,
        chat_messages(instruction, input_text=input_text),
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_for_model(
    *,
    prompts: list[dict[str, Any]],
    model_name: str,
    model_path: str,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_causal_lm_for_training(model_path, use_lora=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rendered_prompts = [
        render_prompt(tokenizer, row["instruction"], row.get("input", ""))
        for row in prompts
    ]
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            prompt_batch = rendered_prompts[start : start + batch_size]
            row_batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            generated = model.generate(**encoded, **generation_kwargs)
            continuation = generated[:, encoded["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for row, response in zip(row_batch, responses):
                records.append(
                    {
                        "model": model_name,
                        "model_path": model_path,
                        "prompt_id": row["prompt_id"],
                        "cluster_id": row.get("cluster_id", "unknown"),
                        "instruction": row["instruction"],
                        "input": row.get("input", ""),
                        "response": response.strip(),
                        "response_words": len(response.split()),
                        "generation_seed": seed,
                        "max_prompt_length": max_prompt_length,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )
    print(f"Generated {len(records)} outputs for {model_name}")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["prompt_id"], []).append(record)

    lines = ["# Model Generations", ""]
    for prompt_id in sorted(grouped, key=lambda value: int(value) if value.isdigit() else value):
        prompt_records = grouped[prompt_id]
        first = prompt_records[0]
        lines.extend(
            [
                f"## Prompt {prompt_id}",
                "",
                first["instruction"].strip(),
                "",
            ]
        )
        for record in prompt_records:
            lines.extend(
                [
                    f"### {record['model']}",
                    "",
                    record["response"].strip(),
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model outputs for a prompt manifest.")
    parser.add_argument("--prompts-file", type=Path, default=Path("data/test_prompts_10.txt"))
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="NAME=PATH entries, e.g. SFT=Qwen/Qwen2.5-1.5B-Instruct CPO=outputs/checkpoints/cpo",
    )
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/generations/generations.jsonl"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/generations/generations.md"))
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = parse_prompt_file(args.prompts_file)
    model_specs = parse_model_specs(args.models)
    validate_model_specs(model_specs)
    all_records: list[dict[str, Any]] = []
    for model_name, model_path in model_specs:
        all_records.extend(
            generate_for_model(
                prompts=prompts,
                model_name=model_name,
                model_path=model_path,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )
    write_jsonl(args.output_jsonl, all_records)
    write_markdown(args.output_md, all_records)
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
