from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cpo_trl.data import load_training_rows
from cpo_trl.cpo_trainer import CPOConfig, CPOLossComputer
from cpo_trl.eval import (
    collate_mismatched_unary_batch,
    collate_unary_batch,
    encode_unary,
    sequence_logps,
    sequence_logps_with_token_kl,
)
from cpo_trl.finite import assert_finite_gradients, assert_finite_loss
from cpo_trl.peft import load_causal_lm_for_training, lora_settings_from_config
from cpo_trl.sampling import CPOPairAwareBatchSampler
from common import add_common_args, parse_with_config


def cpo_state_payload(loss_computer: CPOLossComputer, *, global_step: int, epoch: float) -> dict[str, object]:
    return {
        "global_step": global_step,
        "epoch": epoch,
        "loss_computer": loss_computer.state_dict(),
    }


def save_cpo_state(path: Path, loss_computer: CPOLossComputer, *, global_step: int, epoch: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cpo_state_payload(loss_computer, global_step=global_step, epoch=epoch), indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_training_state(
    path: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    global_step: int,
    micro_step: int,
    epoch: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "micro_step": micro_step,
            "epoch": epoch,
        },
        path,
    )


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return torch.zeros((), dtype=values.dtype, device=values.device)


class MetricWindow:
    """Accumulate per-micro-batch metrics over one optimizer (accumulation) window.

    Logged values are then the example-weighted mean over the micro-batches that
    actually formed the gradient step, instead of a single last-micro-batch snapshot.
    """

    _KEYS = ("rew_pos", "rew_neg", "lr_pos", "lr_neg", "lp_pos", "lp_neg")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n_examples = 0
        self.unary_sum = 0.0
        self.kl_sum = 0.0
        self.num_pairs = 0
        self.pair_loss_sum = 0.0
        self.pair_capacity = 0
        self.baseline_count = 0
        self._acc = {key: [0.0, 0] for key in self._KEYS}

    def add(
        self,
        *,
        n_examples: int,
        unary: float,
        kl: float,
        num_pairs: int,
        pair_loss: float,
        pair_capacity: int,
        baseline_count: int,
        rewards: torch.Tensor,
        logratios: torch.Tensor,
        logps: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> None:
        self.n_examples += n_examples
        self.unary_sum += unary * n_examples
        self.kl_sum += kl * n_examples
        self.num_pairs += num_pairs
        self.pair_loss_sum += pair_loss * num_pairs
        self.pair_capacity += pair_capacity
        self.baseline_count += baseline_count
        for key, values, mask in (
            ("rew_pos", rewards, pos_mask), ("rew_neg", rewards, neg_mask),
            ("lr_pos", logratios, pos_mask), ("lr_neg", logratios, neg_mask),
            ("lp_pos", logps, pos_mask), ("lp_neg", logps, neg_mask),
        ):
            count = int(mask.sum().item())
            if count:
                self._acc[key][0] += float(values[mask].sum().item())
                self._acc[key][1] += count

    def _mean(self, key: str) -> float:
        total, count = self._acc[key]
        return total / count if count else 0.0

    @property
    def unary(self) -> float:
        return self.unary_sum / self.n_examples if self.n_examples else 0.0

    @property
    def kl(self) -> float:
        return self.kl_sum / self.n_examples if self.n_examples else 0.0

    @property
    def pair(self) -> float:
        return self.pair_loss_sum / self.num_pairs if self.num_pairs else 0.0

    def loss(self, alpha: float, kl_coef: float) -> float:
        return (1.0 - alpha) * self.unary + alpha * self.pair + kl_coef * self.kl

    def rew_pos(self) -> float: return self._mean("rew_pos")
    def rew_neg(self) -> float: return self._mean("rew_neg")
    def lr_pos(self) -> float: return self._mean("lr_pos")
    def lr_neg(self) -> float: return self._mean("lr_neg")
    def lp_pos(self) -> float: return self._mean("lp_pos")
    def lp_neg(self) -> float: return self._mean("lp_neg")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--beta", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--alpha", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--lambda-desirable", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--lambda-undesirable", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--kl-coef", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--z-baseline",
        choices=("token_kl", "same_completion_logratio", "kto_mismatched_logratio"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--z-momentum", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--pair-aware-batching", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--cluster-sampling", default=argparse.SUPPRESS)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=argparse.SUPPRESS)
    args = parse_with_config(parser)
    if not hasattr(args, "beta"):
        args.beta = 0.1
    if not hasattr(args, "alpha"):
        args.alpha = 0.3
    if not hasattr(args, "lambda_desirable"):
        args.lambda_desirable = 1.0
    if not hasattr(args, "lambda_undesirable"):
        args.lambda_undesirable = 1.0
    if not hasattr(args, "kl_coef"):
        args.kl_coef = 0.0
    if not hasattr(args, "z_baseline"):
        args.z_baseline = "token_kl"
    if args.z_baseline not in {"token_kl", "same_completion_logratio", "kto_mismatched_logratio"}:
        raise ValueError(f"unsupported z_baseline: {args.z_baseline}")
    if not hasattr(args, "z_momentum"):
        args.z_momentum = 0.9
    if not hasattr(args, "pair_aware_batching"):
        args.pair_aware_batching = True
    if not hasattr(args, "cluster_sampling"):
        args.cluster_sampling = "proportional"
    if not hasattr(args, "resume_from_checkpoint"):
        args.resume_from_checkpoint = None

    from transformers import AutoTokenizer, get_scheduler, set_seed

    set_seed(args.seed)
    rows = sorted(
        load_training_rows(args.train_file, "cpo"),
        key=lambda row: (str(row["prompt_id"]), str(row["cluster_id"]), not bool(row["label"])),
    )
    resume_checkpoint = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    policy_model_path = resume_checkpoint if resume_checkpoint is not None else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora_settings = lora_settings_from_config(args)
    model = load_causal_lm_for_training(
        policy_model_path,
        use_lora=args.use_lora,
        create_lora=resume_checkpoint is None,
        lora_settings=lora_settings,
    )
    ref_model = load_causal_lm_for_training(args.model_name_or_path, use_lora=False)
    ref_model.requires_grad_(False)
    ref_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)
    encoded = [encode_unary(tokenizer, row, args.max_seq_length) for row in rows]
    diagnostics_path = Path(args.output_dir) / "cpo_diagnostics.jsonl"
    write_jsonl_record(
        diagnostics_path,
        {
            "event": "startup",
            "rows": len(rows),
            "max_seq_length": args.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "pair_aware_batching": bool(args.pair_aware_batching),
            "cluster_sampling": args.cluster_sampling,
            "z_baseline": args.z_baseline,
            "alpha": args.alpha,
            "beta": args.beta,
            "kl_coef": args.kl_coef,
        },
    )
    batch_sampler = None
    if args.pair_aware_batching:
        batch_sampler = CPOPairAwareBatchSampler(
            encoded,
            batch_size=args.per_device_train_batch_size,
            seed=args.seed,
            cluster_sampling=args.cluster_sampling,
        )
        sampler_record = {
            "event": "sampler",
            "pair_aware_batching": True,
            "cluster_sampling": args.cluster_sampling,
            "eligible_prompt_groups": batch_sampler.stats.eligible_prompt_groups,
            "skipped_prompt_groups": batch_sampler.stats.skipped_prompt_groups,
            "sampler_batches": batch_sampler.stats.batches,
            "sampler_pairs": batch_sampler.stats.pairs,
            "sampler_cluster_counts": batch_sampler.stats.cluster_counts,
            "sampler_cluster_pairs": batch_sampler.stats.cluster_pairs,
            "paired_rows_seen": batch_sampler.stats.paired_rows_seen,
            "unpaired_rows_seen": batch_sampler.stats.unpaired_rows_seen,
            "unpaired_rows_used": batch_sampler.stats.unpaired_rows_used,
            "unpaired_rows_dropped": batch_sampler.stats.unpaired_rows_dropped,
            "dataset_rows_used": batch_sampler.stats.dataset_rows_used,
            "dataset_coverage": batch_sampler.stats.dataset_coverage,
            "sampler_cluster_paired_rows": batch_sampler.stats.cluster_paired_rows,
            "sampler_cluster_unary_rows": batch_sampler.stats.cluster_unary_rows,
            "baseline_ready_batches": batch_sampler.stats.baseline_ready_batches,
            "baseline_ready_rate": batch_sampler.stats.baseline_ready_rate,
        }
        print(sampler_record)
        write_jsonl_record(diagnostics_path, sampler_record)
    dataloader_kwargs = {
        "collate_fn": lambda batch: collate_unary_batch(tokenizer, batch),
    }
    if batch_sampler is None:
        dataloader_kwargs.update({"batch_size": args.per_device_train_batch_size, "shuffle": False})
    else:
        dataloader_kwargs["batch_sampler"] = batch_sampler
    dataloader = DataLoader(encoded, **dataloader_kwargs)
    if len(dataloader) == 0:
        raise ValueError("CPO dataloader has no batches; check pair-aware batching eligibility or dataset size")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_micro_steps = max(1, math.ceil(len(dataloader) * args.num_train_epochs))
    total_steps = max(1, math.ceil(total_micro_steps / args.gradient_accumulation_steps))
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps or int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_computer = CPOLossComputer(
        CPOConfig(
            beta=args.beta,
            alpha=args.alpha,
            lambda_desirable=args.lambda_desirable,
            lambda_undesirable=args.lambda_undesirable,
            kl_coef=args.kl_coef,
            z_momentum=args.z_momentum,
        )
    )
    global_step = 0
    start_micro_step = 0
    state_path = Path(args.output_dir) / "cpo_state.json"
    if resume_checkpoint is not None:
        checkpoint_cpo_state = resume_checkpoint / "cpo_state.json"
        checkpoint_training_state = resume_checkpoint / "training_state.pt"
        if not checkpoint_cpo_state.exists():
            raise FileNotFoundError(f"missing CPO state for resume: {checkpoint_cpo_state}")
        if not checkpoint_training_state.exists():
            raise FileNotFoundError(f"missing optimizer/scheduler state for resume: {checkpoint_training_state}")
        state = json.loads(checkpoint_cpo_state.read_text(encoding="utf-8"))
        loss_computer.load_state_dict(dict(state.get("loss_computer", {})))
        training_state = torch.load(checkpoint_training_state, map_location="cpu")
        optimizer.load_state_dict(training_state["optimizer"])
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(training_state["scheduler"])
        global_step = int(training_state["global_step"])
        start_micro_step = int(training_state["micro_step"])
        print(f"resumed CPO training from {resume_checkpoint} at optimizer step {global_step}")
    metrics_path = Path(args.output_dir) / "train_metrics.jsonl"
    grouped_metrics_path = Path(args.output_dir) / "train_metrics_grouped.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    total_observed_pairs = 0
    zero_pair_micro_steps = 0
    no_baseline_micro_steps = 0
    logged_optimizer_steps = 0
    metric_window = MetricWindow()
    while micro_step < total_micro_steps:
        sampler_epoch = micro_step // max(1, len(dataloader))
        if batch_sampler is not None:
            batch_sampler.set_epoch(sampler_epoch)
        for batch in dataloader:
            if micro_step >= total_micro_steps:
                break
            if micro_step < start_micro_step:
                micro_step += 1
                continue
            accumulation_start = micro_step - (micro_step % args.gradient_accumulation_steps)
            accumulation_end = min(accumulation_start + args.gradient_accumulation_steps, total_micro_steps)
            accumulation_size = accumulation_end - accumulation_start
            if micro_step % args.gradient_accumulation_steps == 0:
                metric_window.reset()
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            policy_logps, ref_logps, kl_values = sequence_logps_with_token_kl(
                model,
                ref_model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_start"],
            )
            baseline_values = None
            baseline_cluster_ids = None
            update_z = True
            baseline_count = len(batch["cluster_ids"])
            if args.z_baseline == "same_completion_logratio":
                baseline_values = (policy_logps - ref_logps).detach()
                baseline_cluster_ids = [str(cluster_id) for cluster_id in batch["cluster_ids"]]
            elif args.z_baseline == "kto_mismatched_logratio":
                mismatched = collate_mismatched_unary_batch(
                    tokenizer,
                    prompt_texts=[str(prompt) for prompt in batch["prompt_texts"]],
                    completions=[str(completion) for completion in batch["completions"]],
                    prompt_ids=[str(prompt_id) for prompt_id in batch["prompt_ids"]],
                    cluster_ids=[str(cluster_id) for cluster_id in batch["cluster_ids"]],
                    max_length=args.max_seq_length,
                )
                if mismatched is None:
                    update_z = False
                    baseline_count = 0
                else:
                    mismatched = {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in mismatched.items()
                    }
                    with torch.no_grad():
                        policy_baseline_logps = sequence_logps(
                            model,
                            mismatched["input_ids"],
                            mismatched["attention_mask"],
                            mismatched["response_start"],
                        )
                        ref_baseline_logps = sequence_logps(
                            ref_model,
                            mismatched["input_ids"],
                            mismatched["attention_mask"],
                            mismatched["response_start"],
                        )
                    baseline_values = (policy_baseline_logps - ref_baseline_logps).detach()
                    baseline_cluster_ids = [str(cluster_id) for cluster_id in mismatched["cluster_ids"]]
                    baseline_count = len(baseline_cluster_ids)
            metrics = loss_computer(
                policy_logps=policy_logps,
                ref_logps=ref_logps,
                labels=batch["labels"],
                prompt_ids=[str(prompt_id) for prompt_id in batch["prompt_ids"]],
                cluster_ids=batch["cluster_ids"],
                baseline_values=baseline_values,
                baseline_cluster_ids=baseline_cluster_ids,
                kl_values=kl_values,
                pair_indices=batch["pair_indices"] if args.pair_aware_batching else None,
                update_z=update_z,
            )
            total_observed_pairs += metrics.num_pairs
            if metrics.num_pairs == 0:
                zero_pair_micro_steps += 1
            if baseline_count == 0:
                no_baseline_micro_steps += 1
            mb_logratios = policy_logps.detach() - ref_logps.detach()
            mb_pos_mask = batch["labels"].bool()
            metric_window.add(
                n_examples=int(batch["labels"].numel()),
                unary=metrics.unary_loss.item(),
                kl=metrics.kl_loss.item(),
                num_pairs=metrics.num_pairs,
                pair_loss=metrics.pair_loss.item(),
                pair_capacity=max(1, len(batch["cluster_ids"]) // 2),
                baseline_count=baseline_count,
                rewards=args.beta * mb_logratios,
                logratios=mb_logratios,
                logps=policy_logps.detach(),
                pos_mask=mb_pos_mask,
                neg_mask=~mb_pos_mask,
            )
            loss = metrics.loss / accumulation_size
            assert_finite_loss(loss)
            loss.backward()
            micro_step += 1
            current_epoch = micro_step / max(1, len(dataloader))
            should_step = micro_step % args.gradient_accumulation_steps == 0 or micro_step == total_micro_steps
            if should_step:
                assert_finite_gradients(model.named_parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                current_learning_rate = scheduler.get_last_lr()[0]
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                logged_optimizer_steps += 1
                if global_step % args.logging_steps == 0:
                    # Metrics are the example-weighted mean over the accumulation
                    # window that formed this optimizer step (not a last-micro-batch snapshot).
                    window_loss = metric_window.loss(args.alpha, args.kl_coef)
                    window_unary = metric_window.unary
                    window_pair = metric_window.pair
                    window_kl = metric_window.kl
                    window_num_pairs = metric_window.num_pairs
                    positive_reward_mean = metric_window.rew_pos()
                    negative_reward_mean = metric_window.rew_neg()
                    positive_logratio_mean = metric_window.lr_pos()
                    negative_logratio_mean = metric_window.lr_neg()
                    positive_logp_mean = metric_window.lp_pos()
                    negative_logp_mean = metric_window.lp_neg()
                    reward_margin = positive_reward_mean - negative_reward_mean
                    logratio_margin = positive_logratio_mean - negative_logratio_mean
                    pair_capacity = max(1, metric_window.pair_capacity)
                    pair_coverage = window_num_pairs / pair_capacity
                    window_baseline_count = metric_window.baseline_count
                    zero_pair_rate = zero_pair_micro_steps / max(1, micro_step - start_micro_step)
                    no_baseline_rate = no_baseline_micro_steps / max(1, micro_step - start_micro_step)
                    record = {
                        "step": global_step,
                        "epoch": current_epoch,
                        "loss": window_loss,
                        "unary_loss": window_unary,
                        "pair_loss": window_pair,
                        "kl_loss": window_kl,
                        "kl_coef": args.kl_coef,
                        "z_baseline": args.z_baseline,
                        "pair_aware_batching": bool(args.pair_aware_batching),
                        "cluster_sampling": args.cluster_sampling,
                        "baseline_count": window_baseline_count,
                        "num_pairs": window_num_pairs,
                        "pair_coverage": pair_coverage,
                        "total_observed_pairs": total_observed_pairs,
                        "zero_pair_micro_steps": zero_pair_micro_steps,
                        "no_baseline_micro_steps": no_baseline_micro_steps,
                        "zero_pair_micro_step_rate": zero_pair_rate,
                        "no_baseline_micro_step_rate": no_baseline_rate,
                        "reward_margin": reward_margin,
                        "positive_reward_mean": positive_reward_mean,
                        "negative_reward_mean": negative_reward_mean,
                        "logratio_margin": logratio_margin,
                        "positive_logratio_mean": positive_logratio_mean,
                        "negative_logratio_mean": negative_logratio_mean,
                        "positive_logp_mean": positive_logp_mean,
                        "negative_logp_mean": negative_logp_mean,
                        "grad_norm": float(grad_norm.detach().cpu().item()),
                        "learning_rate": current_learning_rate,
                        "z_k": metrics.z_k,
                        "cluster_counts": metrics.cluster_counts,
                    }
                    grouped_record = {
                        "step": global_step,
                        "epoch": current_epoch,
                        "objective": {
                            "loss": window_loss,
                            "unary_loss": window_unary,
                            "pair_loss": window_pair,
                            "kl_loss": window_kl,
                            "kl_coef": args.kl_coef,
                            "alpha": args.alpha,
                            "beta": args.beta,
                        },
                        "pairwise_signal": {
                            "num_pairs": window_num_pairs,
                            "pair_capacity": pair_capacity,
                            "pair_coverage": pair_coverage,
                            "total_observed_pairs": total_observed_pairs,
                            "zero_pair_micro_steps": zero_pair_micro_steps,
                            "zero_pair_micro_step_rate": zero_pair_rate,
                        },
                        "cluster_baseline": {
                            "z_baseline": args.z_baseline,
                            "z_k": metrics.z_k,
                            "cluster_counts": metrics.cluster_counts,
                            "baseline_count": window_baseline_count,
                            "no_baseline_micro_steps": no_baseline_micro_steps,
                            "no_baseline_micro_step_rate": no_baseline_rate,
                        },
                        "preference_movement": {
                            "reward_margin": reward_margin,
                            "positive_reward_mean": positive_reward_mean,
                            "negative_reward_mean": negative_reward_mean,
                            "logratio_margin": logratio_margin,
                            "positive_logratio_mean": positive_logratio_mean,
                            "negative_logratio_mean": negative_logratio_mean,
                            "positive_logp_mean": positive_logp_mean,
                            "negative_logp_mean": negative_logp_mean,
                        },
                        "optimization": {
                            "grad_norm": float(grad_norm.detach().cpu().item()),
                            "learning_rate": current_learning_rate,
                            "max_grad_norm": args.max_grad_norm,
                        },
                        "sampler": {
                            "pair_aware_batching": bool(args.pair_aware_batching),
                            "cluster_sampling": args.cluster_sampling,
                        },
                        "run_state": {
                            "global_step": global_step,
                            "micro_step": micro_step,
                            "total_micro_steps": total_micro_steps,
                            "logged_optimizer_steps": logged_optimizer_steps,
                            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                        },
                    }
                    print(record)
                    write_jsonl_record(metrics_path, record)
                    write_jsonl_record(grouped_metrics_path, grouped_record)
                if global_step % args.save_steps == 0:
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    save_cpo_state(
                        checkpoint_dir / "cpo_state.json",
                        loss_computer,
                        global_step=global_step,
                        epoch=current_epoch,
                    )
                    save_training_state(
                        checkpoint_dir / "training_state.pt",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        micro_step=micro_step,
                        epoch=current_epoch,
                    )
                    save_cpo_state(state_path, loss_computer, global_step=global_step, epoch=current_epoch)
                    save_training_state(
                        Path(args.output_dir) / "training_state.pt",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        micro_step=micro_step,
                        epoch=current_epoch,
                    )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    final_epoch = micro_step / max(1, len(dataloader))
    write_jsonl_record(
        diagnostics_path,
        {
            "event": "final",
            "global_step": global_step,
            "micro_step": micro_step,
            "logged_optimizer_steps": logged_optimizer_steps,
            "total_micro_steps": total_micro_steps,
            "total_observed_pairs": total_observed_pairs,
            "zero_pair_micro_steps": zero_pair_micro_steps,
            "no_baseline_micro_steps": no_baseline_micro_steps,
            "zero_pair_micro_step_rate": zero_pair_micro_steps / max(1, micro_step - start_micro_step),
            "no_baseline_micro_step_rate": no_baseline_micro_steps / max(1, micro_step - start_micro_step),
            "z_k": dict(loss_computer.z.values),
            "cluster_counts": dict(loss_computer.z.counts),
        },
    )
    save_cpo_state(state_path, loss_computer, global_step=global_step, epoch=final_epoch)
    save_training_state(
        Path(args.output_dir) / "training_state.pt",
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        micro_step=micro_step,
        epoch=final_epoch,
    )


if __name__ == "__main__":
    main()
