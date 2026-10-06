# @author: Marcio Lopes
#
# EXPERIMENT 1 -- SFL viability WITH host-memory offloading.
#
# Same shard and same ladder as baseline_oom_multiclient_SFL.py; the difference
# is that the model states (FP32 master weights, Adam moments and gradients)
# live in host RAM and the working weights are streamed in and out of the GPU,
# so the resident footprint is dominated by the activations.
#
# This is the offload-enabled control R1.3 asks for, under matched assumptions:
# same shard, same workload, same concurrency ladder, same GPU.
#
# Launch from the PARENT directory (see 00_review/__init__.py):
#   cd runLINDA/case_studies/cds/baselines
#   python 00_review/baseline_oom_multiclient_SFL_offloading.py --device_idx 0 --device_name RTX3090
#   python 00_review/baseline_oom_multiclient_SFL_offloading.py --device_idx 2 --device_name A40
#
# Host RAM, not VRAM, is the constraint that usually binds here: each replica
# holds master + moments + gradient + shadow in host memory. The engine logs its
# footprint at construction, and a host exhaustion is recorded as `oom_host`
# instead of crashing the sweep.

import argparse
import sys

import torch

sys.path.append('../../../../')
from utils.huggingface_token import HF_TOKEN
from utils.model_factory import ModelFactory

from exp1_common import now_tag, run_cell, host_snapshot
from offload_engine import CPUOffloadEngine, estimate_host_gib

_MASTER_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def parse_args():
    ap = argparse.ArgumentParser(
        description="EXP1: SFL concurrency ceiling on one GPU, host-memory offloading")
    ap.add_argument("--device_idx", type=int, required=True,
                    help="CUDA index of the server GPU")
    ap.add_argument("--device_name", type=str, required=True,
                    help="label for the results, e.g. RTX3090 or A40")
    ap.add_argument("--pool", type=str, default="",
                    help="label of the GPU pool, e.g. 'SFL1' or 'SFL2'")
    ap.add_argument("--model_name", type=str, default="google/gemma-2b")
    ap.add_argument("--start_layer", type=int, default=1)
    ap.add_argument("--end_layer", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--hidden_size", type=int, default=2048)
    ap.add_argument("--n_clients", type=int, default=4,
                    help="how many concurrent clients this run targets; the "
                         "sweep over N is external, one process per level")
    ap.add_argument("--lr", type=float, default=5e-6)
    # ---- offloading ----
    ap.add_argument("--master_dtype", type=str, default="fp32",
                    choices=["fp32", "bf16"],
                    help="precision of the CPU master weights and Adam moments")
    ap.add_argument("--pin_memory", type=int, default=1,
                    help="1: pin the CPU masters (faster PCIe, non-swappable RAM)")
    ap.add_argument("--offload_params", type=int, default=1,
                    help="1: stream the heavy module weights in and out of the GPU")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device_idx}")
    now = now_tag()
    pool = args.pool or f"SFL_{args.device_name}"
    cell = f"SFL_{args.device_name}_offload"
    master_dtype = _MASTER_DTYPES[args.master_dtype]
    scope = "full" if int(args.offload_params) else "optimizer_only"

    cfg = {
        "cell": cell,
        "technique": "SFL",
        "pool": pool,
        "device_name": args.device_name,
        "device_idx": args.device_idx,
        "offload": True,
        "offload_scope": scope,
        "master_dtype": args.master_dtype,
        "concurrency": "concurrent",   # every forward before any backward, no threads
        "model_name": args.model_name,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "n_blocks": args.end_layer - args.start_layer,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
        "n_clients": args.n_clients,
        "act_dtype": torch.bfloat16,
        "now": now,
        "run_id": f"b_exp1_{cell}_{now}",
    }

    # ---- the only part that differs from the GPU-resident twin ----
    def build_session(client_id):
        model = ModelFactory.get_model_shard(
            model_name=args.model_name, model_type="llm",
            start_layer=args.start_layer, end_layer=args.end_layer,
            is_first=False, is_last=True,
            hf_token=HF_TOKEN, device=device)
        model.train()

        if client_id == 1:
            est = estimate_host_gib(model, master_dtype, args.n_clients)
            free = host_snapshot()["ram_free_gb"]
            print(f"    host RAM needed for {args.n_clients} replica(s): "
                  f"{est:.2f} GiB (master {args.master_dtype}) | free now: {free:.2f} GiB")
            if est > free:

                print(f"    [warning] the estimate exceeds the free RAM; if the "
                      f"OOM killer fires, nothing is recorded")

        engine = CPUOffloadEngine(
            model=model, device=device, lr=args.lr,
            offload_optimizer=True,
            offload_params=bool(int(args.offload_params)),
            master_dtype=master_dtype,
            pin_memory=bool(int(args.pin_memory)),
            clip_max_norm=1.0,
            label=f"{cell}-c{client_id}")
        return {"client_id": client_id, "model": model, "optimizer": engine}

    def step_session(session):
        # the engine clips on the host gradients and pushes the updated weights back
        session["optimizer"].step()
        session["optimizer"].zero_grad()

    def release_session(session):
        session["optimizer"].close()
    # ---------------------------------------------------------------

    run_cell(cfg, build_session, step_session, release_session)


if __name__ == "__main__":
    main()
