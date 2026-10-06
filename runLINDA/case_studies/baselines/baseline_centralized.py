# baseline_centralized.py
# @author: Marcio Lopes
# Centralized training baseline for Gemma-2B + Alpaca (single GPU, no split learning).
# Usage:
#   cd runLINDA/case_studies/cds/baselines
#   python baseline_centralized.py --device cuda:0 --new_size 0.1 --rounds 5 --batch_size 2 --lr 5e-6

import argparse
import datetime
import gc
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import tqdm
import transformers
from torch.utils.data import DataLoader

sys.path.append('../../../..')
from utils.datasets.alpaca import AlpacaDataset, collate_fn
from utils.datasets.data_partitioner import DataPartitioner
from transformers import AutoModelForCausalLM
from utils.huggingface_token import HF_TOKEN
from utils.linda_logger import logger
from utils import helpers


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized Training Baseline (Gemma-2B + Alpaca)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--new_size", type=float, default=0.1,
                        help="Fraction of Alpaca dataset to use (0.0-1.0)")
    parser.add_argument("--eval_fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Logged for comparison with federated runs")
    parser.add_argument("--data_dir", type=str, default="~/.cache/huggingface")
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    """Builds train/eval dataloaders using the same pipeline as node_agent."""
    eval_indices, train_pool_indices, universe_indices = \
        DataPartitioner.get_global_eval_and_train_pool(
            dataset_name="tatsu-lab/alpaca",
            new_size=args.new_size,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            cache_dir=args.data_dir,
        )

    logger.info(
        f"[DATA] Universe={len(universe_indices)} | "
        f"Eval={len(eval_indices)} | Train={len(train_pool_indices)}"
    )

    # Centralized: single client uses the entire train pool
    train_dataset = AlpacaDataset(
        tokenizer_name=args.model_name,
        split="train",
        indices=train_pool_indices,
        cache_dir=args.data_dir,
    )

    val_dataset = AlpacaDataset(
        tokenizer_name=args.model_name,
        split="train",
        indices=eval_indices,
        cache_dir=args.data_dir,
    )

    pin = args.device.startswith("cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin,
    )

    logger.info(
        f"[DATA] Train samples={len(train_dataset)} batches={len(train_loader)} | "
        f"Eval samples={len(val_dataset)} batches={len(val_loader)}"
    )

    return train_loader, val_loader


def build_model(args, device):
    """Loads the FULL Gemma-2B model directly (no sharding wrapper)."""
    logger.info(f"[Centralized] Loading {args.model_name} (full model, FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    if device.type == "cuda":
        model.gradient_checkpointing_enable()
        logger.info("[Centralized] Gradient checkpointing enabled.")
    model = model.to(device)
    return model


def train_one_round(model, optimizer, train_loader, device, args, round_idx, run_id, criterion):
    model.train()

    snapshot_round_start = helpers.get_local_resource_snapshot(device)
    helpers.log_snapshots(
        node_id="centralized",
        run_id=run_id,
        snapshot_start=snapshot_round_start,
        snapshot_end=snapshot_round_start,
        description="Round_Start",
        r=round_idx,
        i=0,
    )

    for i, batch in enumerate(tqdm.tqdm(train_loader, desc=f"[Train | Round {round_idx}]")):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        optimizer.zero_grad()

        t_start = time.perf_counter()

        outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        use_cache=False, return_dict=True)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max_norm)
        optimizer.step()

        helpers.sync_device(device)

        t_end = time.perf_counter()

        loss_val = loss.item()

        # Snapshot after first iteration (optimizer states now allocated)
        if i % 100 == 0:
            snapshot_after_first = helpers.get_local_resource_snapshot(device)
            helpers.log_snapshots(
                node_id="centralized",
                run_id=run_id,
                snapshot_start=snapshot_round_start,
                snapshot_end=snapshot_after_first,
                description="After_Step_and_loss",
                r=round_idx,
                i=i,
            )

        helpers.log_training_metrics(
            run_id=run_id,
            client_id="centralized",
            group_id=0,
            round_idx=round_idx,
            iteration_idx=i,
            model_type="llm",
            metrics={"loss": loss_val},
            alpha=args.alpha,
            seed=args.seed,
        )

        helpers.log_performance_metric(
            node_id="centralized",
            client_id="centralized",
            seed=args.seed,
            run_id=run_id,
            metric_name="T_comp_train_step",
            duration=(t_end - t_start),
            round_idx=round_idx,
            batch_idx=i,
            req_id=None,
            delay=None,
        )

    # Snapshot at round end
    snapshot_round_end = helpers.get_local_resource_snapshot(device)
    helpers.log_snapshots(
        node_id="centralized",
        run_id=run_id,
        snapshot_start=snapshot_round_start,
        snapshot_end=snapshot_round_end,
        description="Round_End",
        r=round_idx,
        i=len(train_loader),
    )


@torch.no_grad()
def evaluate(model, val_loader, device, criterion):
    model.eval()

    total_loss_sum = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in tqdm.tqdm(val_loader, desc="[Eval]"):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        use_cache=False, return_dict=True)
        logits = outputs.logits

        loss_sum, correct, n_tokens = helpers.compute_eval_batch_metrics(
            model_type="llm",
            output=logits,
            labels=labels,
            criterion=criterion,
            device=device,
        )

        total_loss_sum += loss_sum
        total_correct += correct
        total_tokens += n_tokens

    if total_tokens > 0:
        avg_loss = total_loss_sum / total_tokens
        avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        accuracy = total_correct / total_tokens
    else:
        avg_loss, avg_ppl, accuracy = 0.0, 0.0, 0.0

    model.train()
    return {
        "avg_loss": avg_loss,
        "avg_ppl": avg_ppl,
        "next_token_acc": accuracy,
        "num_tokens": total_tokens,
    }


def main():
    args = parse_args()
    run_id = f"baseline_centralized_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

    set_global_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"[Centralized] Device: {device} | Run: {run_id}")

    initial_snapshot = helpers.get_local_resource_snapshot(device)
    logger.info(f"[Centralized] Initial snapshot: {initial_snapshot}")

    # --- Data ---
    train_loader, val_loader = build_dataloaders(args)

    # --- Model ---
    snapshot_before = helpers.get_local_resource_snapshot(device)
    model = build_model(args, device)


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    snapshot_after = helpers.get_local_resource_snapshot(device)
    logger.info(f"[Centralized] Model loaded. Before: {snapshot_before} | After: {snapshot_after}")
    helpers.log_snapshots(
        node_id="centralized",
        run_id=run_id,
        snapshot_start=snapshot_before,
        snapshot_end=snapshot_after,
        description="Model_Full_Load", r=-1, i=-1,
    )
    # --- Save experiment config at startup ---
    experiment_config = {
        "meta": {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": args.seed,
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(device),
            "initial_system_snapshot": initial_snapshot,
        },
        "model": {
            "name": args.model_name,
            "type": "llm",
            "training_mode": "FFT",
            "total_layers": model.config.num_hidden_layers,
        },
        "dataset": {
            "name": "tatsu-lab/alpaca",
            "new_size": args.new_size,
            "eval_fraction": args.eval_fraction,
            "train_samples": len(train_loader.dataset),
            "train_batches": len(train_loader),
            "eval_samples": len(val_loader.dataset),
            "eval_batches": len(val_loader),
        },
        "hyperparameters": {
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "clip_max_norm": args.clip_max_norm,
            "rounds": args.rounds,
            "alpha": args.alpha,
        },
        "command": " ".join(sys.argv),
    }
    helpers.save_experiment_metadata("centralized", run_id, experiment_config)

    # --- Training Loop ---
    for r in range(args.rounds):
        logger.info(f"=== Round {r}/{args.rounds - 1} ===")

        # LR decay (same schedule as LINDA LLM)
        if r > 0:
            args.lr *= 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
        logger.info(f"[Round {r}] LR: {args.lr}")

        t_train_start = time.perf_counter()
        train_one_round(model, optimizer, train_loader, device, args, r, run_id, criterion)
        t_train_end = time.perf_counter()
        train_time = t_train_end - t_train_start

        helpers.log_performance_metric(
            node_id="centralized",
            client_id="centralized",
            seed=args.seed,
            run_id=run_id,
            metric_name="T_round_train",
            duration=train_time,
            round_idx=r,
            batch_idx=len(train_loader),
            req_id=None,
            delay=None,
        )

        # --- Evaluation ---
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        t_eval_start = time.perf_counter()
        eval_metrics = evaluate(model, val_loader, device, criterion)
        t_eval_end = time.perf_counter()
        eval_time = t_eval_end - t_eval_start

        logger.info(
            f"[Round {r}] Eval: loss={eval_metrics['avg_loss']:.4f} | "
            f"ppl={eval_metrics['avg_ppl']:.4f} | "
            f"acc={eval_metrics['next_token_acc']:.4f} | "
            f"tokens={eval_metrics['num_tokens']} | "
            f"train_time={train_time:.1f}s | eval_time={eval_time:.1f}s"
        )

        helpers.log_evaluation_summary(
            run_id=run_id,
            round_idx=r,
            model_type="llm",
            group_id=0,
            clients_list=["centralized"],
            metrics=eval_metrics,
            train_time=train_time,
            eval_time=eval_time,
            seed=args.seed,
            alpha=args.alpha,
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # --- Save Metadata ---
    metadata = {
        "meta": {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": args.seed,
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(device),
            "hw_name": helpers.get_local_device_name(device),
        },
        "training": {
            "baseline_type": "centralized",
            "model_name": args.model_name,
            "total_layers": model.config.num_hidden_layers,
            "rounds": args.rounds,
            "batch_size": args.batch_size,
            "lr_initial": args.lr,
            "weight_decay": args.weight_decay,
            "clip_max_norm": args.clip_max_norm,
            "new_size": args.new_size,
            "eval_fraction": args.eval_fraction,
            "alpha": args.alpha,
        },
    }
    helpers.save_experiment_metadata("centralized", run_id, metadata)

    logger.info(f"[Centralized] Training finished. Results in results/{run_id}/")


if __name__ == "__main__":
    main()