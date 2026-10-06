# @author: Marcio Lopes
#
# EXPERIMENT 1 -- SFL viability, GPU-resident (no host-memory offloading).
#
# Standard SFL puts the whole post-client model on a single server: the client
# keeps `--start_layer` layers and the server holds everything from there to the
# LM head. Each concurrent client gets its own replica of that shard, so the
# server footprint grows linearly with the number of clients.
#
# This is the twin of baseline_oom_multiclient_SFL_offloading.py: the two differ
# only in build_session() and step_session(), so `diff` between them is exactly
# the offloading change under test.
#
# Launch from the PARENT directory (see 00_review/__init__.py):
#   cd runLINDA/case_studies/cds/baselines
#   python 00_review/baseline_oom_multiclient_SFL.py --device_idx 0 --device_name RTX3090
#   python 00_review/baseline_oom_multiclient_SFL.py --device_idx 2 --device_name A40

import argparse
import sys

import torch

sys.path.append('../../../../')
from utils.huggingface_token import HF_TOKEN
from utils.model_factory import ModelFactory

from exp1_common import now_tag, run_cell


def parse_args():
    ap = argparse.ArgumentParser(
        description="EXP1: SFL concurrency ceiling on one GPU, GPU-resident")
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
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device_idx}")
    now = now_tag()
    pool = args.pool or f"SFL_{args.device_name}"
    cell = f"SFL_{args.device_name}_nooffload"

    cfg = {
        "cell": cell,
        "technique": "SFL",
        "pool": pool,
        "device_name": args.device_name,
        "device_idx": args.device_idx,
        "offload": False,
        "offload_scope": "none",
        "master_dtype": "",
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

    # ---- the only part that differs from the offloading twin ----
    def build_session(client_id):
        model = ModelFactory.get_model_shard(
            model_name=args.model_name, model_type="llm",
            start_layer=args.start_layer, end_layer=args.end_layer,
            is_first=False, is_last=True,
            hf_token=HF_TOKEN, device=device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        return {"client_id": client_id, "model": model, "optimizer": optimizer}

    def step_session(session):
        torch.nn.utils.clip_grad_norm_(session["model"].parameters(), max_norm=1.0)
        session["optimizer"].step()
        session["optimizer"].zero_grad(set_to_none=True)

    def release_session(session):
        session["optimizer"].zero_grad(set_to_none=True)
    # -------------------------------------------------------------

    run_cell(cfg, build_session, step_session, release_session)


if __name__ == "__main__":
    main()
