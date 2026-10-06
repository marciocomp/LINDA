# @author: Marcio Lopes
"""
Shared measurement code for the Experiment 1 viability cells.

Experiment 1 asks a memory question, not a latency one, so every cell runs in a
single process on synthetic activations: the peak footprint depends on tensor
shapes, not on tensor values, and dropping the data pipeline and the network
removes two sources of noise from the measurement.

What the cell emulates
----------------------
A server that was not memory-bound would admit every client at once and keep
training. So the N replicas are all resident and every forward is issued before
any backward, which keeps N logit tensors and N activation sets alive at the
same time -- the regime the per-client model of Table V accounts for, and the
one baseline_oom_multigpus_HSFL.py already uses.

No threads: CUDA memory is per process, not per thread, so N threads on one GPU
draw from the same pool and the persistent terms are identical either way.
Threads would only add non-deterministic allocation order, and with it a
run-to-run spread in the reserved figure.

Two stages, each its own loop over the clients
----------------------------------------------
1. LOAD + WARM-UP -- for n in 1..N: build replica n, then run one full
              iteration on it. Building and exercising in the same loop keeps
              each replica's footprint real before the next is built, so an OOM
              lands on the client that does not fit. The warm-up iteration is
              what makes the first optimizer step happen; nothing is measured
              here.
2. MEASURE -- peak counters reset, then forward for all, backward for all, step
              for all. Every term of Table V is resident at once.

One process measures one client count. The sweep is external, so every level
starts on a virgin allocator:

    for n in 1 2 3 4; do python 00_review/<cell>.py ... --n_clients $n; done

Failures are recorded with the phase and the resource that ran out, so a cell
limited by host RAM is never read as a policy limit.
"""

import csv
import datetime
import gc
import json
import os

import torch

from utils.helpers import (get_local_resource_snapshot, log_snapshots,
                           get_block_checkpoint_cost_gib, get_unit_block_cost_gib,
                           get_head_cost_gib, get_logits_cost_gib, results_path)

CSV_FIELDS = [
    "cell", "technique", "pool", "device_name", "device_idx", "gpu_name",
    "offload", "offload_scope", "master_dtype", "concurrency",
    "n_clients", "peak_alloc_gib", "peak_reserved_gib", "budget_gib",
    "host_ram_used_gib", "host_ram_free_gib",
    "status", "fail_phase", "fail_resource",
    "pred_per_client_gib", "pred_total_gib", "run_id", "timestamp",
]

STATUS_OK = "ok"
PHASE_LOAD = "load"
PHASE_COMPUTE = "compute"
RESOURCE_VRAM = "vram"
RESOURCE_HOST = "host"


def gib(x_bytes):
    return float(x_bytes) / (1024 ** 3)


def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")


def classify_failure(exc):
    """
    Tells a VRAM exhaustion from a host RAM exhaustion.

    Needed because the offload cells move the model states into host RAM, where
    running out surfaces as MemoryError or as a plain RuntimeError from the
    pinned-memory allocator -- neither of which is a torch.cuda.OutOfMemoryError.
    Without this, an offload cell would die in a traceback instead of recording
    its ceiling. Returns None for anything that is not a memory failure, so the
    caller re-raises instead of inventing a ceiling.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return RESOURCE_VRAM
    if isinstance(exc, MemoryError):
        return RESOURCE_HOST
    text = str(exc).lower()
    if "cuda" in text and "out of memory" in text:
        return RESOURCE_VRAM
    for marker in ("cannot allocate memory", "out of memory", "pinned",
                   "cuda host", "bad_alloc", "cudahostalloc"):
        if marker in text:
            return RESOURCE_HOST
    return None


def predicted_cost(model_name, batch_size, seq_len, n_blocks, n_clients,
                   checkpointing=True):
    """
    Per-client tail footprint from the paper's memory model.

    `checkpointing=True` is the operative one: ModelFactory calls
    gradient_checkpointing_enable() on CUDA, so the measured run always uses the
    checkpointed block cost. The non-checkpointed figure is reported alongside
    only because the earlier script printed that one, which made its printed
    prediction inconsistent with what it measured.
    """
    block = (get_block_checkpoint_cost_gib(model_name, batch_size, seq_len)
             if checkpointing
             else get_unit_block_cost_gib(model_name, batch_size, seq_len))
    per_client = (n_blocks * block
                  + get_head_cost_gib(model_name)
                  + get_logits_cost_gib(model_name, batch_size, seq_len))
    return per_client, per_client * n_clients


def host_snapshot():
    return get_local_resource_snapshot(torch.device("cpu"))


def make_dummy_batch(batch_size, seq_len, hidden_size, device, dtype):
    """Synthetic smashed activations and labels, shaped as the real ones."""
    x = torch.randn(batch_size, seq_len, hidden_size,
                    device=device, dtype=dtype, requires_grad=True)
    y = torch.randint(0, 1000, (batch_size, seq_len),
                      device=device, dtype=torch.long)
    return x, y


def calculate_loss(criterion, output, labels):
    logits = output[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return criterion(logits.view(-1, logits.size(-1)), shift_labels.view(-1))


def mem_report(device, tag):
    torch.cuda.synchronize(device)
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(f"    [{tag}] free={gib(free_b):.2f}/{gib(total_b):.2f} GiB | "
          f"alloc={gib(torch.cuda.memory_allocated(device)):.2f} | "
          f"reserved={gib(torch.cuda.memory_reserved(device)):.2f}")


def append_csv(row):
    path = results_path(row["run_id"], "exp1_viability.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    return path


def save_json(cell_dir, name, payload):
    os.makedirs(cell_dir, exist_ok=True)
    path = os.path.join(cell_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
    return path


class _Stop(Exception):
    """Internal: unwinds to the single cleanup path once a stage has failed."""


def run_cell(cfg, build_session, step_session, release_session):
    """
    Measures one Experiment 1 cell at a single client count.

    Two stages, each its own loop over the clients:

        STAGE 1  LOAD + WARM-UP
                          for n in 1..N: build replica n, then run one full
                          iteration on it, so its footprint is real before the
                          next replica is built
        STAGE 2  MEASURE  peak counters reset, then
                            for n in 1..N: forward  (graph kept)
                            for n in 1..N: backward
                            for n in 1..N: optimizer step
                          Every forward is issued before any backward, so the N
                          logit tensors and activation sets are alive at once --
                          the regime the per-client model of Table V accounts
                          for. Splitting the measured iteration into three
                          passes is what keeps that property.

    An OOM in any stage stops the cell and is recorded with the stage, the
    resource that ran out and the client index that triggered it.

    The N sweep is external, one process per level, so every level starts on a
    virgin allocator:

        for n in 1 2 3 4; do python 00_review/<cell>.py ... --n_clients $n; done
    """
    device = torch.device(f"cuda:{cfg['device_idx']}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    budget = props.total_memory / (1024 ** 3)
    n_target = cfg["n_clients"]

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    cell_dir = results_path(cfg["run_id"])

    pred_per_client, _ = predicted_cost(
        cfg["model_name"], cfg["batch_size"], cfg["seq_len"], cfg["n_blocks"], 1)
    pred_nockpt, _ = predicted_cost(
        cfg["model_name"], cfg["batch_size"], cfg["seq_len"], cfg["n_blocks"], 1,
        checkpointing=False)

    print("=" * 72)
    print(f"EXP1 cell '{cfg['cell']}'  |  N = {n_target} concurrent client(s)")
    print(f"  device      : cuda:{cfg['device_idx']} {props.name} | {budget:.2f} GiB")
    print(f"  shard       : blocks {cfg['start_layer']}-{cfg['end_layer']} "
          f"({cfg['n_blocks']}) + LM head")
    print(f"  workload    : batch {cfg['batch_size']} | seq {cfg['seq_len']} | FFT")
    print(f"  regime      : all forwards before any backward, no threads")
    print(f"  offload     : {cfg['offload_scope']}"
          + (f" | master {cfg['master_dtype']}" if cfg["offload"] else ""))
    print(f"  predicted   : {pred_per_client:.2f} GiB/client (checkpointed) | "
          f"{pred_nockpt:.2f} GiB/client (no checkpointing)")
    print("=" * 72)

    snapshot_start = get_local_resource_snapshot(device)
    host_start = host_snapshot()

    sessions = []
    fail_stage = fail_resource = fail_at = None
    peak_alloc = peak_reserved = None

    def _fail(stage, client_idx, exc):
        resource = classify_failure(exc)
        if resource is None:
            raise exc
        print(f"\n[X] OOM during {stage.upper()} at client {client_idx} "
              f"on {resource.upper()} -- {type(exc).__name__}: {exc}")
        try:
            mem_report(device, f"cuda:{cfg['device_idx']} @failure")
        except Exception:
            pass
        return stage, resource, client_idx

    try:
        # ---- STAGE 1: LOAD + WARM-UP ---------------------------------------

        print(f"\n[STAGE 1/2] LOAD + WARM-UP -- {n_target} replica(s)")
        for n in range(1, n_target + 1):
            try:
                sessions.append(build_session(n))
            except BaseException as exc:                  # noqa: BLE001
                fail_stage, fail_resource, fail_at = _fail(PHASE_LOAD, n, exc)
                raise _Stop()
            try:
                s = sessions[n - 1]
                x, y = make_dummy_batch(cfg["batch_size"], cfg["seq_len"],
                                        cfg["hidden_size"], device, cfg["act_dtype"])
                s["optimizer"].zero_grad()
                calculate_loss(criterion, s["model"](x), y).backward()
                step_session(s)
                del x, y
            except BaseException as exc:                  # noqa: BLE001
                fail_stage, fail_resource, fail_at = _fail(PHASE_COMPUTE, n, exc)
                raise _Stop()
            print(f"    client {n}/{n_target} loaded and warmed up")
        gc.collect()

        # ---- STAGE 3: MEASURE ----------------------------------------------
        print(f"\n[STAGE 2/2] MEASURE -- {n_target} concurrent client(s)")
        torch.cuda.reset_peak_memory_stats(device)
        live = []
        try:
            for n in range(1, n_target + 1):
                s = sessions[n - 1]
                x, y = make_dummy_batch(cfg["batch_size"], cfg["seq_len"],
                                        cfg["hidden_size"], device, cfg["act_dtype"])
                s["optimizer"].zero_grad()
                live.append((s, calculate_loss(criterion, s["model"](x), y)))
        except BaseException as exc:                      # noqa: BLE001
            fail_stage, fail_resource, fail_at = _fail(PHASE_COMPUTE, len(live) + 1, exc)
            raise _Stop()

        try:
            for n in range(1, n_target + 1):
                live[n - 1][1].backward()
        except BaseException as exc:                      # noqa: BLE001
            fail_stage, fail_resource, fail_at = _fail(PHASE_COMPUTE, n, exc)
            raise _Stop()

        try:
            for n in range(1, n_target + 1):
                step_session(live[n - 1][0])
        except BaseException as exc:                      # noqa: BLE001
            fail_stage, fail_resource, fail_at = _fail(PHASE_COMPUTE, n, exc)
            raise _Stop()

        torch.cuda.synchronize(device)
        peak_alloc = gib(torch.cuda.max_memory_allocated(device))
        peak_reserved = gib(torch.cuda.max_memory_reserved(device))
        del live

    except _Stop:
        pass

    gc.collect()
    host_now = host_snapshot()
    host_used = host_start["ram_free_gb"] - host_now["ram_free_gb"]
    ok = fail_stage is None
    status = STATUS_OK if ok else f"oom_{fail_stage}_{fail_resource}"

    if ok:
        print(f"\n[N={n_target}] OK  peak_alloc={peak_alloc:.2f}  "
              f"peak_reserved={peak_reserved:.2f} / {budget:.2f} GiB  "
              f"|  host_used={host_used:.2f} GiB")

    csv_path = append_csv(dict(
        cfg, gpu_name=props.name, budget_gib=round(budget, 2), n_clients=n_target,
        peak_alloc_gib=round(peak_alloc, 3) if ok else "",
        peak_reserved_gib=round(peak_reserved, 3) if ok else "",
        host_ram_used_gib=round(host_used, 3),
        host_ram_free_gib=round(host_now["ram_free_gb"], 3),
        status=status, fail_phase=fail_stage or "", fail_resource=fail_resource or "",
        pred_per_client_gib=round(pred_per_client, 3),
        pred_total_gib=round(pred_per_client * n_target, 3),
        timestamp=datetime.datetime.now().isoformat(timespec="seconds")))
    log_snapshots(node_id=cfg["cell"], run_id=cfg["run_id"],
                  snapshot_start=snapshot_start,
                  snapshot_end=get_local_resource_snapshot(device),
                  r=n_target, i=0, description=status)

    result = dict(cfg)
    result.pop("act_dtype", None)
    result.update({
        "gpu_name": props.name,
        "budget_gib": round(budget, 2),
        "status": status,
        "fail_stage": fail_stage,
        "fail_resource": fail_resource,
        "fail_at_client": fail_at,
        "clients_sustained": (fail_at - 1) if fail_at else n_target,
        "peak_alloc_gib": round(peak_alloc, 3) if ok else None,
        "peak_reserved_gib": round(peak_reserved, 3) if ok else None,
        "host_ram_used_gib": round(host_used, 3),
        "pred_per_client_gib": round(pred_per_client, 3),
        "pred_per_client_gib_no_checkpointing": round(pred_nockpt, 3),
    })
    path = save_json(cell_dir, f"{cfg['cell']}_N{n_target}_{cfg['now']}.json", result)

    while sessions:
        spent = sessions.pop()
        release_session(spent)
        spent.pop("optimizer", None)
        spent.pop("model", None)
        del spent
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print(f"cell              : {cfg['cell']}  N={n_target}")
    print("outcome           : " + (
        "completed" if ok else
        f"OOM in {fail_stage} at client {fail_at} on {fail_resource} "
        f"-> {fail_at - 1} client(s) sustained"))
    print(f"json              : {path}")
    print(f"csv               : {csv_path}")
    print("=" * 72)
    return result
