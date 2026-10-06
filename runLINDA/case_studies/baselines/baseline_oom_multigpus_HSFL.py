# @author: Marcio Lopes
#
# Static HSFL with a MULTI-GPU chain, single logical cluster.
#
# Purpose: show that giving Static HSFL more GPUs does not raise its concurrency
# ceiling. The chain relieves the transformer blocks, never the tail: the tail
# node must host LM Head + Logit Tensors for every concurrent client, and that
# cost is per-client and irreducible.
#
# Chain mirrors LINDA's strong site (3x RTX 3090 + 1x A40 as the tail), so the
# hardware pool is the same one LINDA uses; only the orchestration differs.
#
# Usage (from runLINDA/case_studies/cds/baselines/):
#   python baseline_oom_multigpus_HSFL.py
#   python baseline_oom_multigpus_HSFL.py --variant minimal_tail
#   python baseline_oom_multigpus_HSFL.py --devices 5,4,3,0 --clients 4

import argparse
import gc
import json
import os
import sys
import datetime

import torch

sys.path.append('../../../../')
from utils.huggingface_token import HF_TOKEN
from utils.model_factory import ModelFactory
from utils.helpers import (get_local_resource_snapshot, log_snapshots,
                           get_block_checkpoint_cost_gib, get_head_cost_gib,
                           get_logits_cost_gib, results_path)

MODEL_NAME = "google/gemma-2b"
HIDDEN_SIZE = 2048
BATCH_SIZE = 16
SEQ_LEN = 512
TOTAL_LAYERS = 18
T1_LAYERS = 1  # Tier 1 keeps embeddings + block 0, as in every experiment

NOW = datetime.datetime.now().strftime("%Y%m%d%H%M")
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)


VARIANTS = {
    "linda_mirror": [5, 5, 5, 2],
    "minimal_tail": [6, 6, 4, 1],
    "two_a40": [16, 1],
}


def run_id_for(variant_name):
    """The run id, shared by the snapshot CSVs and the summary JSON, so both
    land in the same results directory instead of the JSON sitting loose in
    the launch directory."""
    return f"b_exp1_HSFL_multigpu_{variant_name}_{NOW}"


def gib(x_bytes):
    return float(x_bytes) / (1024 ** 3)


def mem_report(device, tag):
    torch.cuda.synchronize(device)
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(f"    [{tag}] free={gib(free_b):.2f} / {gib(total_b):.2f} GiB | "
          f"allocated={gib(torch.cuda.memory_allocated(device)):.2f} | "
          f"reserved={gib(torch.cuda.memory_reserved(device)):.2f}")


def calculate_loss(output, labels):
    logits = output[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return criterion(logits.view(-1, logits.size(-1)), shift_labels.view(-1))


def build_chain(device_ids, block_counts):
    """Maps block ranges onto the chain. The last hop carries the LM head."""
    chain, start = [], T1_LAYERS
    for i, (dev_id, n_blocks) in enumerate(zip(device_ids, block_counts)):
        end = start + n_blocks
        chain.append({
            "device": torch.device(f"cuda:{dev_id}"),
            "device_id": dev_id,
            "start": start,
            "end": end,
            "is_last": (i == len(device_ids) - 1),
        })
        start = end
    if start != TOTAL_LAYERS:
        raise ValueError(f"block counts sum to {start - T1_LAYERS}, "
                         f"expected {TOTAL_LAYERS - T1_LAYERS}")
    return chain


def predicted_tail_cost(n_blocks_tail, n_clients):
    """Analytical tail footprint, from the paper's memory model."""
    per_client = (n_blocks_tail * get_block_checkpoint_cost_gib(MODEL_NAME, BATCH_SIZE, SEQ_LEN)
                  + get_head_cost_gib(MODEL_NAME)
                  + get_logits_cost_gib(MODEL_NAME, BATCH_SIZE, SEQ_LEN))
    return per_client, per_client * n_clients


def forward_client(shards, chain):
    """
    Forward pass along the chain, with detached hops as in split execution.

    The graphs are RETAINED and returned: every concurrent client keeps its
    activations and logit tensor resident until its backward runs, which is the
    condition the per-client memory model of Table V accounts for.
    """
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=chain[0]["device"],
                    dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN),
                           device=chain[-1]["device"], dtype=torch.long)

    inputs, outputs = [], []
    cur = x
    for shard, hop in zip(shards, chain):
        cur = cur.to(hop["device"]).detach().requires_grad_(True)
        inputs.append(cur)
        out = shard(cur)
        outputs.append(out)
        cur = out

    loss = calculate_loss(outputs[-1], labels)
    return {"inputs": inputs, "outputs": outputs, "loss": loss}


def backward_client(state, chain):
    """Backward in reverse topological order, gradient handed to the previous hop."""
    grad = None
    for i in reversed(range(len(chain))):
        if i == len(chain) - 1:
            state["loss"].backward()
        else:
            state["outputs"][i].backward(grad.to(chain[i]["device"]))
        grad = state["inputs"][i].grad
    return float(state["loss"].detach().cpu())


def run(device_ids, block_counts, n_clients, variant_name):
    chain = build_chain(device_ids, block_counts)

    print("=" * 72)
    print(f"Static HSFL, multi-GPU chain -- variant '{variant_name}'")
    print(f"Model {MODEL_NAME} | batch {BATCH_SIZE} | seq {SEQ_LEN} | FFT")
    for hop in chain:
        props = torch.cuda.get_device_properties(hop["device"])
        role = "TAIL (blocks + LM head + logits)" if hop["is_last"] else "chain"
        print(f"  cuda:{hop['device_id']:<2} {props.name:<24} "
              f"blocks {hop['start']:>2}-{hop['end']:<2} ({hop['end']-hop['start']}) "
              f"| {props.total_memory/(1024**3):.2f} GiB | {role}")

    n_tail_blocks = chain[-1]["end"] - chain[-1]["start"]
    per_client, total = predicted_tail_cost(n_tail_blocks, n_clients)
    tail_budget = torch.cuda.get_device_properties(chain[-1]["device"]).total_memory / (1024 ** 3)
    print(f"\nAnalytical tail cost: {per_client:.2f} GiB/client "
          f"-> {total:.2f} GiB for {n_clients} clients, budget {tail_budget:.2f} GiB")
    print(f"Prediction: {'OOM' if total > tail_budget else 'fits'}")
    print("=" * 72)

    snapshots_start = {h["device_id"]: get_local_resource_snapshot(h["device"]) for h in chain}
    sessions = []
    oom_at, oom_device, peak_by_level = None, None, {}
    client_id = 0

    try:
        for client_id in range(1, n_clients + 1):
            print(f"\n[Level {client_id}] allocating replicas along the chain...")
            shards, optims = [], []
            for hop in chain:
                torch.cuda.set_device(hop["device"])
                shard = ModelFactory.get_model_shard(
                    model_name=MODEL_NAME, model_type="llm",
                    start_layer=hop["start"], end_layer=hop["end"],
                    is_first=False, is_last=hop["is_last"],
                    hf_token=HF_TOKEN, device=hop["device"])
                shard.train()
                shards.append(shard)
                optims.append(torch.optim.AdamW(shard.parameters(), lr=5e-6))
            sessions.append((shards, optims))

            for hop in chain:
                torch.cuda.reset_peak_memory_stats(hop["device"])

            print(f"[Level {client_id}] forward pass for {client_id} concurrent client(s)...")
            live = [forward_client(s, chain) for s, _ in sessions]

            print(f"[Level {client_id}] backward pass...")
            for st in live:
                backward_client(st, chain)
            for _, optims_c in sessions:
                for opt in optims_c:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            print(f"[Level {client_id}] PEAK with {client_id} concurrent client(s):")
            for hop in chain:
                torch.cuda.synchronize(hop["device"])
                peak = gib(torch.cuda.max_memory_allocated(hop["device"]))
                reserved = gib(torch.cuda.max_memory_reserved(hop["device"]))
                budget = torch.cuda.get_device_properties(hop["device"]).total_memory / (1024 ** 3)
                tag = "  <-- TAIL" if hop["is_last"] else ""
                print(f"    cuda:{hop['device_id']} peak_alloc={peak:6.2f} "
                      f"peak_reserved={reserved:6.2f} / {budget:.2f} GiB{tag}")
                peak_by_level.setdefault(client_id, {})[hop["device_id"]] = round(peak, 2)

            del live
            gc.collect()
            print(f"[Level {client_id}] OK -- {client_id} concurrent client(s) completed")

            for hop in chain:
                log_snapshots(node_id=f"cuda{hop['device_id']}_HSFL_multigpu",
                              run_id=run_id_for(variant_name),
                              snapshot_start=snapshots_start[hop["device_id"]],
                              snapshot_end=get_local_resource_snapshot(hop["device"]),
                              description="concurrent_clients",
                              r=client_id, i=0)

    except torch.cuda.OutOfMemoryError:
        oom_at = client_id
        # attribute the failure to the most saturated device
        oom_device = min(chain, key=lambda h: torch.cuda.mem_get_info(h["device"])[0])["device_id"]
        print(f"\n[X] OOM with {client_id} concurrent client(s), "
              f"tightest device cuda:{oom_device}")
        for hop in chain:
            mem_report(hop["device"], f"cuda:{hop['device_id']} @OOM")

    result = {
        "variant": variant_name,
        "devices": device_ids,
        "block_counts": block_counts,
        "clients_requested": n_clients,
        "clients_completed": (oom_at - 1) if oom_at else n_clients,
        "oom_at_client": oom_at,
        "oom_device": oom_device,
        "tail_blocks": n_tail_blocks,
        "tail_cost_per_client_gib": round(per_client, 2),
        "tail_cost_total_gib": round(total, 2),
        "tail_budget_gib": round(tail_budget, 2),
        "peak_allocated_gib_by_level": peak_by_level,
    }

    del sessions
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Static HSFL over a multi-GPU chain")
    ap.add_argument("--devices", type=str, default="5,4,3,0",
                    help="CUDA indices along the chain; last one is the tail")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--variant", type=str, default="linda_mirror",
                    choices=list(VARIANTS.keys()),
                    help="a preset chain from VARIANTS; ignored when --blocks is given")
    ap.add_argument("--blocks", type=str, default="",
                    help="explicit block count per hop, e.g. '4,4,4,4,1'. Overrides "
                         "--variant, so a new chain shape needs no code change. Must "
                         "sum to TOTAL_LAYERS - T1_LAYERS and match --devices in length.")
    ap.add_argument("--label", type=str, default="",
                    help="name for the results directory when --blocks is used; "
                         "defaults to the block list itself")
    args = ap.parse_args()

    device_ids = [int(d) for d in args.devices.split(",")]
    if args.blocks:
        block_counts = [int(b) for b in args.blocks.split(",")]
        variant_name = args.label or "custom_" + "-".join(args.blocks.split(","))
    else:
        block_counts = VARIANTS[args.variant]
        variant_name = args.variant
    if len(device_ids) != len(block_counts):
        raise SystemExit(f"--devices has {len(device_ids)} entries but the chain "
                         f"has {len(block_counts)}: {block_counts}")

    res = run(device_ids, block_counts, args.clients, variant_name)

    run_id = run_id_for(variant_name)
    out_dir = results_path(run_id)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{run_id}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=4)

    print("\n" + "=" * 72)
    print(f"clients completed : {res['clients_completed']} / {res['clients_requested']}")
    print(f"OOM at client     : {res['oom_at_client']} (cuda:{res['oom_device']})"
          if res["oom_at_client"] else "no OOM")
    print(f"saved             : {out}")
    print("=" * 72)
