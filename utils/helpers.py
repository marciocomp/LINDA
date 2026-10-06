# @author: Marcio Lopes
from collections import Counter, defaultdict
from .linda_logger import *
import torch
import psutil
import os
import csv
import time
import pandas as pd
from scipy import stats
import threading
import math
import json
import sys
import platform
import numpy as np
from datetime import datetime
import ctypes
import traceback

_file_locks = defaultdict(threading.Lock)

# ==========================================
# Model Architecture Specifications
# ==========================================
MODEL_SPECS = {
    "google/gemma-2b": {
        "hidden_size": 2048,
        "vocab_size": 256128,
        "num_layers": 18,
        "total_params": 2.51e9,
        "block_params": 110_000_000,
    },
    "google/gemma-7b": {
        "hidden_size": 3072,
        "vocab_size": 256128,
        "num_layers": 28,
        "total_params": 8.5e9,
        "block_params": 264_000_000,
    },

    # Add other models if necessary
}

# Byte size constants
BYTES_FP16 = 2
BYTES_FP32 = 4
BYTES_OPTIMIZER = 8  # AdamW (Momentum + Variance)
BYTES_STATIC_PER_PARAM = BYTES_FP16 + BYTES_FP16 + BYTES_OPTIMIZER
GIB = 1024 ** 3

_device_contexts = {}

PROBE_N_CUDA = 2048
PROBE_N_CPU = 512
PROBE_WARMUP = 5
PROBE_ITERS = 20

PROBE_REPEATS = 3


# Every run writes under this tree. It is resolved from this file, not from the
# working directory, so all processes of a run agree on it wherever they start.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runLINDA", "case_studies", "results")

SHARD_LAYER_PREFIX = "internal_model.layers."

FATAL_EXIT_CODE = 17

FAILURE_COLUMNS = ["timestamp", "node_id", "client_id", "seed", "run_id", "round",
                   "batch", "phase", "kind", "device", "device_name", "layers",
                   "total_gib", "free_now_gib", "held_by_this_process_gib",
                   "held_by_others_gib", "torch_allocated_gib",
                   "torch_reserved_gib", "message"]

def _get_model_spec(model_name):
    """
    Retrieve model specs by name with partial matching
    (e.g. 'google/gemma-2b-it' matches 'gemma-2b').
    """
    name_clean = model_name.lower()

    for key, spec in MODEL_SPECS.items():
        if key in name_clean:
            return spec

    logger.warning(f"Model specs for '{model_name}' not found. Defaulting to 'gemma-2b' baseline.")
    return None

# ==========================================
# Cost Calculation Functions (Updated)
# ==========================================

def get_head_cost_gib(model_name):
    """
        Static cost of the LM Head.
        Formula: Vocab * Hidden * 12 bytes
    """
    spec = _get_model_spec(model_name)
    head_params = spec["vocab_size"] * spec["hidden_size"]
    total_bytes = head_params * BYTES_STATIC_PER_PARAM
    return round(total_bytes / GIB, 4)

def get_logits_cost_gib(model_name, batch_size, seq_len=2048):
    """
    Dynamic cost of logit tensors.
    Formula: B * S * V * 4 bytes (Fwd/Bwd) + 15% Overhead
    """

    spec = _get_model_spec(model_name)
    elements = batch_size * seq_len * spec["vocab_size"]
    # Base (Fwd+Bwd FP16) + 15% Overhead
    total_bytes = (elements * (2 * BYTES_FP16)) #* 1.15
    return round(total_bytes / GIB, 4)

def get_embeddings_cost_gib(model_name, batch_size, seq_len=2048):
    """
    Total cost of the input embeddings layer, typically held at Tier 1: 12 bytes per
    parameter for weights, gradients and optimizer, plus the output tensor.
    """
    spec = _get_model_spec(model_name)

    # Params = Vocab * Hidden
    input_params = spec["vocab_size"] * spec["hidden_size"]
    static_bytes = input_params * BYTES_STATIC_PER_PARAM

    # Shape: [Batch, Seq, Hidden] * 2 bytes (FP16)
    activation_bytes = batch_size * seq_len * spec["hidden_size"] * BYTES_FP16

    return round((static_bytes + activation_bytes) / GIB, 4)

def get_block_checkpoint_cost_gib(model_name, batch_size, seq_len=2048):
    """
    Block cost WITH checkpointing (Tail/Tier 3).
    Formula: (Params * 12) + (InputTensor * 2)
    """

    spec = _get_model_spec(model_name)

    block_static_bytes = spec["block_params"] * BYTES_STATIC_PER_PARAM

    # B * S * H * 2 bytes
    input_tensor_bytes = batch_size * seq_len * spec["hidden_size"] * BYTES_FP16

    return round((block_static_bytes + input_tensor_bytes) / GIB, 4)

def get_unit_block_cost_gib(model_name, batch_size, seq_len=2048):
    """
    Block cost for an intermediate Tier 2 node: (params * 12) + (input tensor * 2 *
    expansion). The expansion factor is 1 because every CUDA shard runs with gradient
    checkpointing, whatever its tier; restore it if that becomes conditional.
    """
    spec = _get_model_spec(model_name)


    block_static_bytes = spec["block_params"] * BYTES_STATIC_PER_PARAM

    input_tensor_bytes = batch_size * seq_len * spec["hidden_size"] * BYTES_FP16

    EXPANSION_FACTOR = 1

    return round((block_static_bytes + (input_tensor_bytes * EXPANSION_FACTOR)) / GIB, 4)

def sync_device(device):
    """
    Drains the CUDA queue so that a perf_counter() reading reflects executed work. It
    synchronizes the whole device, so readings taken while threads share one GPU must be
    aggregated by span, never summed.
    """
    try:
        if isinstance(device, str):
            device = torch.device(device)
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)
    except Exception:
        pass

def payload_nbytes(obj):
    """
    Size in bytes of a message payload, following tensors inside dicts/lists.
    Used to report the per-hop activation/gradient volume.
    """
    try:
        if torch.is_tensor(obj):
            return obj.element_size() * obj.nelement()
        if isinstance(obj, dict):
            return sum(payload_nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(payload_nbytes(v) for v in obj)
    except Exception:
        pass
    return 0

def results_path(run_id, *parts):
    """Where a run writes its files: runLINDA/case_studies/results/{run_id}/."""
    return os.path.join(RESULTS_DIR, str(run_id), *parts)

def load_proxy_map(topology):
    """
    Optional "proxies" section of the topology file, mapping a host IP to the IP of
    the proxy that forwards traffic back to that host. A topology that declares none
    keeps every connection direct.
    """
    if not isinstance(topology, dict):
        return {}
    proxies = topology.get("proxies") or {}
    return {str(host): str(proxy) for host, proxy in proxies.items() if proxy}

def dial_ip(proxy_map, local_ip, target_ip):
    """
    Address to dial to reach target_ip from local_ip. Two processes on the same host
    would talk over loopback, so the connection goes through the host's proxy when the
    topology declares one; hops to another host stay direct.
    """
    if not proxy_map or not local_ip or target_ip != local_ip:
        return target_ip
    return proxy_map.get(target_ip, target_ip)

def log_payload_size(node_id, run_id, client_id, round_idx, hop, payload, batch_idx=0):
    """
    Records the byte volume of one hop, so that bandwidth can be derived as
    bytes / T_link. Only the first batch of a round is recorded, and it goes to
    debug_metrics so the latency sums never pick it up.
    """
    if batch_idx != 0:
        return
    n = payload_nbytes(payload)
    if n <= 0:
        return
    log_debug_metric(node_id=node_id, run_id=run_id, client_id=client_id,
                     round_idx=round_idx, context="PAYLOAD", entity=hop,
                     metric_name="payload_bytes", value=n)

def get_local_device_name(device):
    """
    Returns the actual hardware name (GPU or CPU).
    Supports heterogeneous multi-GPU (e.g. 'cuda:6' -> 'NVIDIA GeForce RTX 3090').
    """
    try:
        if isinstance(device, str):
            device = torch.device(device)

        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            return f"{gpu_name} ({str(device)})"

        elif device.type == 'cpu':
            try:
                # Linux
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            except:
                pass
            return platform.processor() or "CPU (Generic)"

    except Exception as e:
        return f"Unknown Device ({e})"

    return "Unknown Device"

def init_device_context(device):
    """
    Materializes this process's first reservation on `device` and reports its cost. The
    CUDA context does not exist until the process touches the device, and reading
    `mem_get_info` before that overstates the free memory by hundreds of MiB.
    """
    if isinstance(device, str):
        device = torch.device(device)
    key = str(device)
    if key in _device_contexts:
        return _device_contexts[key]

    if device.type != "cuda":
        reading = {"created": False, "reason": "not an accelerator"}
        _device_contexts[key] = reading
        return reading

    if not torch.cuda.is_available():
        reading = {"created": False,
                   "reason": "torch %s reports cuda unavailable (built for CUDA %s)"
                             % (torch.__version__, torch.version.cuda)}
        logger.error("[HW] %s requested but torch.cuda.is_available() is False: %s"
                     % (key, reading["reason"]))
        _device_contexts[key] = reading
        return reading

    try:
        started = time.perf_counter()
        torch.cuda.init()
        probe = torch.empty(1, device=device)
        torch.cuda.synchronize(device)
        del probe
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception as exc:
        reading = {"created": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
        logger.error("[HW] %s context could not be created: %s" % (key, reading["reason"]))
        _device_contexts[key] = reading
        return reading

    reading = {"created": True, "elapsed_s": time.perf_counter() - started,
               "total_gib": total_bytes / GIB, "free_gib": free_bytes / GIB,
               "reserved_gib": (total_bytes - free_bytes) / GIB,
               "gpu_name": torch.cuda.get_device_name(device)}
    logger.info("[HW] %s %s context up in %.2fs | total=%.2f free=%.2f GiB "
                "(context + other processes: %.2f GiB)"
                % (key, reading["gpu_name"], reading["elapsed_s"], reading["total_gib"],
                   reading["free_gib"], reading["reserved_gib"]))
    _device_contexts[key] = reading
    return reading

def measure_device_flops(device, runtime=False):
    """
    Throughput of `device` in TFLOPS, timed rather than declared: a square matmul after
    a warm-up, around explicit synchronization and repeated, so the figure is what is
    available here and now, contention included.
    """
    if isinstance(device, str):
        device = torch.device(device)
    is_cuda = device.type == "cuda"
    n = PROBE_N_CUDA if is_cuda else PROBE_N_CPU
    dtype = torch.bfloat16 if is_cuda else torch.float32

    if is_cuda and not init_device_context(device).get("created"):
        return None

    try:
        a = torch.randn((n, n), device=device, dtype=dtype)
        b = torch.randn((n, n), device=device, dtype=dtype)
        c = torch.empty((n, n), device=device, dtype=dtype)

        for _ in range(PROBE_WARMUP):
            torch.matmul(a, b, out=c)
        if is_cuda:
            torch.cuda.synchronize(device)

        samples = []
        for _ in range(PROBE_REPEATS):
            started = time.perf_counter()
            for _ in range(PROBE_ITERS):
                torch.matmul(a, b, out=c)
            if is_cuda:
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)

        del a, b, c
        if is_cuda:
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.error("[HW] %s throughput could not be measured: %s: %s"
                     % (device, type(exc).__name__, exc))
        return None

    samples.sort()
    # Two operations per multiply-accumulate, n^3 of them per matmul.
    tflops = (2.0 * (n ** 3) * PROBE_ITERS) / samples[len(samples) // 2] / 1e12
    logger.info("[HW] %s throughput: %.2f TFLOPS (median of %d x %d x %d^3, %s)"
                % (device, tflops, PROBE_REPEATS, PROBE_ITERS, n,
                   "bf16" if is_cuda else "fp32"))
    return tflops

def measure_local_resources(node_id, device, runtime=False):
    """
    The resource vector of this node, read from its own hardware: free VRAM when there
    is an accelerator, available RAM otherwise, and throughput timed on the device in
    both cases.
    """
    if isinstance(device, str):
        device = torch.device(device)

    started = time.perf_counter()
    entry = {"node_id": node_id, "device": str(device), "t_probe_s": 0.0}

    if device.type == "cuda":
        context = init_device_context(device)
        if not context.get("created"):
            entry.update({"ram": 0.0, "reason": context.get("reason")})
        else:

            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            held_by_us = torch.cuda.memory_reserved(device)

            probe_started = time.perf_counter()
            entry["flops"] = measure_device_flops(device, runtime=runtime) or 0.0
            entry["t_probe_s"] = time.perf_counter() - probe_started

            entry.update({"ram": (free_bytes + held_by_us) / GIB,
                          "total_gib": total_bytes / GIB,
                          "free_now_gib": free_bytes / GIB,
                          "held_by_this_process_gib": held_by_us / GIB,
                          "reserved_gib": (total_bytes - free_bytes - held_by_us) / GIB,
                          "device_name": context.get("gpu_name")})
    else:
        mem = psutil.virtual_memory()
        probe_started = time.perf_counter()
        entry["flops"] = measure_device_flops(device, runtime=runtime) or 0.0
        entry["t_probe_s"] = time.perf_counter() - probe_started
        entry.update({"ram": mem.available / GIB, "total_gib": mem.total / GIB,
                      "free_now_gib": mem.available / GIB,
                      "held_by_this_process_gib": 0.0,
                      "reserved_gib": (mem.total - mem.available) / GIB,
                      "device_name": platform.processor() or "cpu"})

    entry.setdefault("flops", 0.0)
    entry["t_measure_s"] = time.perf_counter() - started
    logger.info("[HW] %s measured: %.2f GiB allocatable | %.2f TFLOPS "
                "(free now %.2f, held by this process %.2f, by others %.2f) "
                "in %.3fs, of which %.3fs was the probe"
                % (node_id, entry["ram"], entry["flops"],
                   entry.get("free_now_gib", 0.0),
                   entry.get("held_by_this_process_gib", 0.0),
                   entry.get("reserved_gib", 0.0),
                   entry["t_measure_s"], entry["t_probe_s"]))
    return entry

def capture_optimizer_moments(model, optimizer):
    """
    The optimizer state of a shard, keyed by parameter name and copied to host memory.
    Names are the only index that survives the model being dropped, and holding the
    moments on the device would take back the VRAM the drop just released.
    """
    carried = {}
    for name, param in model.named_parameters():
        state = optimizer.state.get(param)
        if not state:
            continue
        carried[name] = {key: (value.detach().to("cpu", copy=True)
                               if torch.is_tensor(value) else value)
                         for key, value in state.items()}
    return carried

def shard_layer_prefix(model_type):
    """
    What precedes the block index in a shard key: the wrapper's path for an LLM, and
    nothing for a vision shard, whose nn.Sequential numbers its blocks from zero.
    """
    return SHARD_LAYER_PREFIX if str(model_type) == "llm" else ""


def carry_optimizer_moments(moments, old_start, new_start, new_end, prefix=SHARD_LAYER_PREFIX):
    """
    The moments that survive a range change, renumbered for the new range. A layer that
    stays keeps its estimates; one that leaves is dropped and starts cold on its new
    node. `prefix` is what precedes the block index in a shard key, empty for the
    positional keys a vision shard uses.
    Returns (moments for the new range, parameters left behind).
    """
    kept, dropped = {}, 0
    for name, state in (moments or {}).items():
        if not name.startswith(prefix):
            kept[name] = state
            continue
        index, _, tail = name[len(prefix):].partition(".")
        if not index.isdigit():
            kept[name] = state
            continue
        absolute = int(old_start) + int(index)
        if not (int(new_start) <= absolute < int(new_end)):
            dropped += 1
            continue
        kept["%s%d.%s" % (prefix, absolute - int(new_start), tail)] = state
    return kept, dropped

def restore_optimizer_moments(model, optimizer, moments):
    """
    Puts carried moments into a freshly built optimizer. A name with no saved state, or
    one whose shape no longer matches, is left cold; `step` is kept, so bias correction
    does not restart.
    """
    if not moments:
        return 0
    restored = 0
    for name, param in model.named_parameters():
        state = moments.get(name)
        if not state:
            continue
        reference = state.get("exp_avg")
        if torch.is_tensor(reference) and tuple(reference.shape) != tuple(param.shape):
            logger.warning("[REALLOC] moments of %s do not match its shape "
                           "(%s vs %s); left cold."
                           % (name, tuple(reference.shape), tuple(param.shape)))
            continue
        rebuilt = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                rebuilt[key] = value
            elif key == "step":
                rebuilt[key] = value.clone()
            else:
                rebuilt[key] = value.to(device=param.device, dtype=param.dtype)
        optimizer.state[param] = rebuilt
        restored += 1
    return restored

def get_local_resource_snapshot(device):
    """
    Collects real-time GPU/CPU telemetry.
    Returns: Dictionary with VRAM (if GPU) or RAM (if CPU) in GB.
    """
    resources = {
        "device": str(device),
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_free_gb": 0.0
    }

    if device.type == 'cuda':
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        resources["vram_free_gb"] = free_bytes / (1024 ** 3)
        resources["vram_total_gb"] = total_bytes / (1024 ** 3)
        resources["gpu_name"] = torch.cuda.get_device_name(device)

    else:

        mem = psutil.virtual_memory()
        resources["ram_total_gb"] = mem.total / (1024 ** 3)
        resources["ram_free_gb"] = mem.available / (1024 ** 3)

    return resources

def get_local_server_id(topology, site_name):
    """
    Helper to find the default Tier 2 entry point for a given site.
    Returns the ID of the first available server in the site's Tier 2 pool.
    """
    site_data = topology['sites'].get(site_name)
    if not site_data:
        return None

    tier2_servers = site_data.get('tier2', {})
    if not tier2_servers:
        return None

    first_server_key = next(iter(tier2_servers))
    return tier2_servers[first_server_key]['id']

def get_all_clients_on_site(topology,site_name):
    site_data = topology['sites'].get(site_name)
    clients = site_data.get('tier1',{})
    return clients

def find_site_of_node(topology, node_id):
    """
    Helper: Finds which 'site_name' (e.g., 'site2') a node_id (e.g., 's2_t1_1') belongs to.
    This ensures we send the config to the correct Cluster Manager.
    """
    for site_name, site_data in topology['sites'].items():
        # Check Tier 1 Devices
        for dev in site_data.get('tier1', {}).values():
            if dev['id'] == node_id: return site_name
        # Check Tier 2 Servers (if needed)
        for srv in site_data.get('tier2', {}).values():
            if srv['id'] == node_id: return site_name
    return None

def resolve_node_ip(topology, site_name, node_id):
    """Looks up a Tier 2 node's IP in the local topology."""
    tier2 = topology['sites'][site_name]['tier2']
    for key, node in tier2.items():
        if node['id'] == node_id:
            return node['ip']
    return "localhost"

def resolve_node_port(topology, site_name, node_id):
    tier2 = topology['sites'][site_name]['tier2']
    for key, node in tier2.items():
        if node['id'] == node_id:
            return node['port'][0]
    return None

def state_dict_stats(sd: dict):
    """
    Compact stats for a state_dict: number of keys, total bytes of the tensors and the
    dtype counts.
    """
    total_bytes = 0
    dtype_counts = Counter()
    tensor_keys = 0

    for _, v in sd.items():
        if torch.is_tensor(v):
            tensor_keys += 1
            total_bytes += v.numel() * v.element_size()
            dtype_counts[str(v.dtype)] += 1

    mb = total_bytes / (1024.0 * 1024.0)
    return {
        "n_keys": len(sd),
        "tensor_keys": tensor_keys,
        "mb": mb,
        "dtype_counts": dtype_counts
    }

def log_state_dict_stats(prefix: str, sd: dict, extra: str = ""):
        st = state_dict_stats(sd)
        top = st["dtype_counts"].most_common(3)
        top_s = ", ".join([f"{k}:{v}" for k, v in top]) if top else "-"
        logger.info(
            f"{prefix} keys={st['n_keys']} tensor_keys={st['tensor_keys']} "
            f"MB={st['mb']:.2f} top_dtypes=[{top_s}] {extra}".strip()
        )

def state_dict_cpu(model, fp16 = False):
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            t = v.detach().cpu()
            if fp16 and t.is_floating_point():
                t = t.half()
            out[k] = t
        else:
            out[k] = v
    return out

def get_weights(model):
    """Extracts weights to CPU."""
    return {k: v.cpu() for k, v in model.state_dict().items()}

def update_weights(model, new_weights, strict=False):
    """Loads weights into the model."""
    keys = model.load_state_dict(new_weights, strict=strict)
    return keys

def apply_local_update(model, msg):
    new_weights = msg[1]
    model.load_state_dict(new_weights, strict=False)

def cast_sd_fp16(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.half()
        else:
            out[k] = v
    return out

def tstats(t: torch.Tensor, name: str = "") -> str:
    """Compact log: shape/dtype/device/norm/absmax/finite."""
    if t is None:
        return f"{name}=None"
    try:
        with torch.no_grad():
            td = t.detach()
            finite = bool(torch.isfinite(td).all().item()) if td.numel() > 0 else True
            # norm/absmax in float32 for numerical stability
            # td_f = td.float()
            # nrm = float(td_f.norm().item()) if td_f.numel() > 0 else 0.0
            # amax = float(td_f.abs().max().item()) if td_f.numel() > 0 else 0.0
            nrm = t.norm().item()
            amax = t.abs().max().item()

            return (f"{name}shape={tuple(td.shape)} dtype={td.dtype} dev={td.device.type} "
                    f"norm={nrm:.3e} absmax={amax:.3e} finite={finite}")
    except Exception as e:
        return f"{name}<stats_error:{e}>"

def calculate_model_norms(model):
    """
    L2 norm of the parameters and of the gradients, if they exist. Measures only, never
    clips.
    """
    total_param_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_param_norm += param_norm.item() ** 2
    total_param_norm = total_param_norm ** 0.5

    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2)
            total_grad_norm += grad_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    return total_param_norm, total_grad_norm

def calculate_state_dict_norm(state_dict):
    """
    Computes accumulated L2 norm of all tensors in a state_dict.
    """
    total_norm = 0.0
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            val = tensor.norm(2).item()
            total_norm += val ** 2
    return total_norm ** 0.5

def log_debug_metric(node_id, run_id, client_id, round_idx, context, entity, metric_name, value):
    """
    Saves one debug metric to a node-specific CSV, under a context such as 'GRAD_CHECK'
    or 'AGG_CHECK'.
    """
    base_dir = results_path(run_id, "debug_metrics")
    os.makedirs(base_dir, exist_ok=True)

    filename = os.path.join(base_dir, f"{node_id}_{run_id}_debug.csv")
    file_exists = os.path.isfile(filename)

    try:
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["timestamp", "node_id", "client_id", "round", "context", "entity", "metric_name", "value"])

            writer.writerow([
                time.time(),
                node_id,
                client_id,
                round_idx,
                context,
                entity,
                metric_name,
                f"{value:.6f}"
            ])
    except Exception as e:
        logger.error(f"Failed to save debug metric for {node_id}: {e}")

def log_snapshots(node_id, run_id, snapshot_start, snapshot_end, description, r, i):
    """
    Records memory consumption (RAM and VRAM) snapshots to CSV.
    Computes the delta (End - Start) to isolate the shard cost.
    """
    try:
        base_dir = results_path(run_id, "snapshots")
        os.makedirs(base_dir, exist_ok=True)

        filename = os.path.join(base_dir, f"{node_id}_snapshots.csv")
        file_exists = os.path.isfile(filename)

        device = snapshot_start.get('device')
        
        ram_total_gb = snapshot_start.get('ram_total_gb',0.0)
        ram_start_gb = snapshot_start.get('ram_free_gb', 0.0)
        ram_end_gb = snapshot_end.get('ram_free_gb', 0.0)
        ram_cost_gb = ram_start_gb - ram_end_gb


        vram_total_gb = snapshot_start.get('vram_total_gb', 0.0)
        vram_start_gb = snapshot_start.get('vram_free_gb', 0.0)
        vram_end_gb = snapshot_end.get('vram_free_gb', 0.0)
        vram_cost_gb = vram_start_gb - vram_end_gb

        row = {
            'node_id': node_id,
            'device': device,
            'run_id': run_id,
            'description': description,
            'round_idx': r,
            'iteration': i,
            # RAM (System Memory)
            'ram_total_gb': round(ram_total_gb, 4),
            'ram_start_gb': round(ram_start_gb, 4),
            'ram_end_gb': round(ram_end_gb, 4),
            'ram_cost_gb': round(ram_cost_gb, 4),

            # VRAM (GPU Memory)
            'vram_total_gb': round(vram_total_gb, 4),
            'vram_start_gb': round(vram_start_gb, 4),
            'vram_end_gb': round(vram_end_gb, 4),
            'vram_cost_gb': round(vram_cost_gb, 4)
        }

        fieldnames = [
            'node_id', 'device', 'run_id', 'description', 'round_idx', 'iteration',
            'ram_total_gb', 'ram_start_gb', 'ram_end_gb', 'ram_cost_gb',
            'vram_total_gb', 'vram_start_gb', 'vram_end_gb', 'vram_cost_gb'
        ]

        with open(filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)


    except Exception as e:
        logger.error(f"Failed to log model shard consumption: {e}")

def log_performance_metric(node_id, client_id ,seed, run_id, metric_name, duration, round_idx, batch_idx, req_id, delay):
    """
    Saves a raw performance metric to CSV with seed and run timestamp support.
    Path: results/performance_metrics/{node_id}_{seed}_{run_timestamp}_metrics.csv
    """
    dir_path = results_path(run_id, "performance_metrics")
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"{node_id}_seed_{seed}_date_{run_id}_metrics.csv")

    file_exists = os.path.isfile(file_path)

    try:
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "node_id", "client_id", "seed", "run_id", "round", "batch", "req_id", "metric", "duration_s", "delay"])

            writer.writerow([
                time.time(),
                node_id,
                client_id,
                seed,
                run_id,
                round_idx,
                batch_idx,
                req_id,
                metric_name,
                f"{duration:.6f}",
                delay
            ])
    except Exception as e:
        print(f"[Logger Error] Failed to log metric: {e}")

def device_state(device):
    """
    What the device holds right now, without probing it. Called after something already
    failed, where a probe would either fail as well or allocate over the state being
    recorded.
    """
    if isinstance(device, str):
        device = torch.device(device)

    state = {"device": str(device), "device_name": "", "total_gib": 0.0,
             "free_now_gib": 0.0, "held_by_this_process_gib": 0.0,
             "held_by_others_gib": 0.0, "torch_allocated_gib": 0.0,
             "torch_reserved_gib": 0.0}
    try:
        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.memory_reserved(device)
            state.update({
                "device_name": torch.cuda.get_device_name(device),
                "total_gib": total_bytes / GIB,
                "free_now_gib": free_bytes / GIB,
                "held_by_this_process_gib": reserved / GIB,
                "held_by_others_gib": (total_bytes - free_bytes - reserved) / GIB,
                "torch_allocated_gib": torch.cuda.memory_allocated(device) / GIB,
                "torch_reserved_gib": reserved / GIB,
            })
        else:
            mem = psutil.virtual_memory()
            state.update({"device_name": platform.processor() or "cpu",
                          "total_gib": mem.total / GIB,
                          "free_now_gib": mem.available / GIB,
                          "held_by_others_gib": (mem.total - mem.available) / GIB})
    except Exception as exc:                      # a dying CUDA context can refuse
        state["device_name"] = f"unreadable: {exc}"
    return state

def log_failure_event(node_id, run_id, seed, client_id, round_idx, batch_idx,
                      device, phase, kind, message, layers=None):
    """
    One row per fatal event, in results/{run_id}/failures/. The columns are the ones
    profiles_*.csv reports at the round boundaries, so a failure can be read against
    the last reading the node took before it.
    """
    dir_path = results_path(run_id, "failures")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{node_id}_failures_{run_id}.csv")
    file_exists = os.path.isfile(file_path)

    state = device_state(device)
    try:
        with open(file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(FAILURE_COLUMNS)
            writer.writerow([
                time.time(), node_id, client_id, seed, run_id, round_idx,
                batch_idx, phase, kind, state["device"], state["device_name"],
                layers if layers is not None else "",
                f"{state['total_gib']:.4f}", f"{state['free_now_gib']:.4f}",
                f"{state['held_by_this_process_gib']:.4f}",
                f"{state['held_by_others_gib']:.4f}",
                f"{state['torch_allocated_gib']:.4f}",
                f"{state['torch_reserved_gib']:.4f}",
                " ".join(str(message).split())[:600],
            ])
    except Exception as exc:
        print(f"[Logger Error] Failed to log failure: {exc}")
    return state


def report_fatal(exc, node_id, run_id, seed, client_id, round_idx, batch_idx,
                 device, phase, layers=None):
    """
    Records the event, prints the traceback and ends the process. Letting the thread die
    instead leaves the site waiting for a forward pass that is not coming, and the run
    occupies the testbed until somebody notices.
    """
    state = log_failure_event(node_id=node_id, run_id=run_id, seed=seed,
                              client_id=client_id, round_idx=round_idx,
                              batch_idx=batch_idx, device=device, phase=phase,
                              kind=type(exc).__name__, message=exc, layers=layers)
    logger.error("[FATAL] %s on %s (client=%s round=%s batch=%s phase=%s): %s",
                 type(exc).__name__, node_id, client_id, round_idx, batch_idx,
                 phase, exc)
    logger.error("[FATAL] device %s: %.2f GiB total, %.2f free, %.2f held here, "
                 "%.2f held by others", state["device"], state["total_gib"],
                 state["free_now_gib"], state["held_by_this_process_gib"],
                 state["held_by_others_gib"])
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(FATAL_EXIT_CODE)


def log_training_metrics(run_id, client_id, group_id, round_idx, iteration_idx,
                         model_type, metrics, alpha, seed):
    """
    Logs the per-batch training metrics to results/{run_id}/training_logs/, by logical
    group. batch_ppl is a noisy per-batch estimate, not the global evaluation
    perplexity.
    """
    dir_path  = results_path(run_id, "training_logs")
    os.makedirs(dir_path, exist_ok=True)

    filename  = f"training_group_id_{group_id}_seed_{seed}_alpha_{alpha}_date_{run_id}.csv"
    file_path = os.path.join(dir_path, filename)

    row = {
        'timestamp': time.time(),
        'round':     round_idx,
        'iteration': iteration_idx,
        'client_id': client_id,
        'alpha':     alpha,
        'loss':      f"{metrics.get('loss', 0.0):.6f}",
    }

    if model_type == 'vision':
        row['feat_mean'] = f"{metrics.get('feat_mean', 0.0):.4f}"
        row['feat_std']  = f"{metrics.get('feat_std',  0.0):.4f}"
        fieldnames = ['timestamp', 'round', 'iteration', 'client_id', 'alpha',
                      'loss', 'feat_mean', 'feat_std']

    elif model_type == 'llm':
        loss_val = metrics.get('loss', 0.0)
        batch_ppl = metrics.get('perplexity', safe_exp(loss_val))
        row['batch_ppl'] = f"{batch_ppl:.4f}"
        fieldnames = ['timestamp', 'round', 'iteration', 'client_id', 'alpha',
                      'loss', 'batch_ppl']

    else:
        fieldnames = ['timestamp', 'round', 'iteration', 'client_id', 'alpha', 'loss']

    lock = _file_locks[file_path]
    with lock:
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[Logger Error] Failed to write training log for {client_id}: {e}")

def log_evaluation_summary(run_id, round_idx, model_type, group_id, clients_list,
                            metrics, train_time, eval_time, seed, alpha):
    """
    Logs the per-group and global evaluation summary to
    results/{run_id}/metrics/global_eval_*.csv, with avg_ppl = exp(avg_loss) for an LLM.
    """
    dir_path = results_path(run_id, "metrics")
    os.makedirs(dir_path, exist_ok=True)

    filename  = f"global_eval_{model_type}_seed_{seed}_alpha_{alpha}_date_{run_id}.csv"
    file_path = os.path.join(dir_path, filename)

    row = {
        'timestamp':     time.time(),
        'round':         round_idx,
        'group_id':      group_id,
        'group_clients': str(clients_list),
        'num_tokens':    metrics.get('num_tokens', 0),
        'avg_loss':      f"{metrics.get('avg_loss', 0.0):.6f}",
        'avg_ppl':       f"{metrics.get('avg_ppl',  0.0):.6f}",
        'train_time':    f"{train_time:.2f}",
        'eval_time':     f"{eval_time:.2f}",
    }

    if model_type == "llm":

        row['next_token_acc'] = f"{metrics.get('next_token_acc', metrics.get('avg_acc', 0.0)):.6f}"
        fieldnames = [
            'timestamp', 'round', 'group_id', 'group_clients', 'num_tokens',
            'avg_loss', 'avg_ppl', 'next_token_acc', 'train_time', 'eval_time',
        ]
    else:
        row['avg_acc'] = f"{metrics.get('avg_acc', 0.0):.6f}"
        fieldnames = [
            'timestamp', 'round', 'group_id', 'group_clients', 'num_tokens',
            'avg_loss', 'avg_ppl', 'avg_acc', 'train_time', 'eval_time',
        ]

    lock = _file_locks[file_path]
    with lock:
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[Logger Error] Failed to write eval summary: {e}")


def log_cka_decision(node_id, run_id, similarity_matrix, node_order, logical_groups=None):
    """
    Saves the CKA similarity matrix and the resulting logical groups, as .npy, .csv and
    .json.
    """
    try:

        dir_path = results_path(run_id, "cka_decision")
        os.makedirs(dir_path, exist_ok=True)

        file_base = f"{node_id}_{run_id}_cka"

        npy_path = os.path.join(dir_path, f"{file_base}_matrix.npy")
        np.save(npy_path, similarity_matrix)

        csv_path = os.path.join(dir_path, f"{file_base}_matrix.csv")

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)

            headers = ['Node_ID'] + node_order
            writer.writerow(headers)

            for i, row in enumerate(similarity_matrix):
                row_vals = [f"{val:.6f}" for val in row]
                row_data = [node_order[i]] + row_vals
                writer.writerow(row_data)

        if logical_groups:
            json_path = os.path.join(dir_path, f"{file_base}_groups.json")

            decision_data = {
                "timestamp": run_id,
                "node_order": node_order,
                "logical_groups": logical_groups
            }

            with open(json_path, 'w') as f:
                json.dump(decision_data, f, indent=4)

        logger.info(f"[CKA] Decision data saved successfully to {dir_path}")

    except Exception as e:
        logger.error(f"[CKA] Failed to log CKA decision: {e}")

def save_experiment_metadata(node_id, run_id, metadata):
    """
    Saves a complete experiment configuration snapshot to JSON.
    Path: results/configs/experiment_config_{run_id}.json
    """
    dir_path = results_path(run_id, "configs")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{node_id}_experiment_config_{run_id}.json")

    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        logger.info(f"[Config] Experiment metadata saved to {file_path}")
    except Exception as e:
        logger.error(f"[Config] Failed to save metadata: {e}")

def generate_statistical_summary(results_dir="../../results/performance_metrics"):
    """
    Reads all metric CSVs, groups by Seed/Round/Metric, and generates a statistical report.
    Computes: Mean, StdDev, and Confidence Interval (95%).
    """
    if not os.path.exists(results_dir):
        print("Metrics directory not found.")
        return

    all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir)
                 if f.endswith("_metrics.csv") and "FINAL_STATISTICAL_SUMMARY" not in f]

    if not all_files:
        print("No metric files found.")
        return

    print(f"Consolidating {len(all_files)} log files...")

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list: return

    full_df = pd.concat(df_list, ignore_index=True)

    required_cols = {'metric', 'round', 'seed', 'duration_s'}
    if not required_cols.issubset(full_df.columns):
        print(f"Missing CSV columns. Expected: {required_cols}, Found: {full_df.columns}")
        return


    per_seed_df = full_df.groupby(['metric', 'round', 'seed'])['duration_s'].mean().reset_index()

    summary = per_seed_df.groupby(['metric', 'round'])['duration_s'].agg(
        mean='mean',
        std='std',
        count='count',
        sem='sem'
    ).reset_index()

    # 95% Confidence Interval
    confidence = 0.95

    def calc_ci(row):
        if row['count'] < 2: return 0.0
        return row['sem'] * stats.t.ppf((1 + confidence) / 2., row['count'] - 1)

    summary['ci95_margin'] = summary.apply(calc_ci, axis=1)
    summary['ci95_lower'] = summary['mean'] - summary['ci95_margin']
    summary['ci95_upper'] = summary['mean'] + summary['ci95_margin']

    output_path = os.path.join(results_dir, "FINAL_STATISTICAL_SUMMARY.csv")
    summary.to_csv(output_path, index=False)
    print(f"Statistical report saved to: {output_path}")

# ============================================================
# Generation Eval Logging (LLM)
# ============================================================

def log_generation_eval_summary(run_id,
                               model_type,
                               seed,
                               alpha,
                               round_idx,
                               group_id,
                               mode,
                               subset_size,
                               max_new_tokens,
                               max_prompt_tokens,
                               temperature,
                               top_p,
                               avg_em,
                               avg_rouge_l,
                               gen_time_s):
    """
    Saves a CSV summary of generation metrics (EM / ROUGE-L) per round/group/mode.
    Path: ../../results/{run_id}/metrics/gen_eval_{model_type}_seed_{seed}_alpha_{alpha}_date_{run_id}.csv
    """
    dir_path = results_path(run_id, "metrics")
    os.makedirs(dir_path, exist_ok=True)

    filename = f"gen_eval_{model_type}_seed_{seed}_alpha_{alpha}_date_{run_id}.csv"
    file_path = os.path.join(dir_path, filename)

    row = {
        "timestamp": time.time(),
        "round": round_idx,
        "group_id": group_id,
        "mode": mode,
        "subset_size": subset_size,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature is not None else "",
        "top_p": top_p if top_p is not None else "",
        "avg_em": f"{avg_em:.6f}",
        "avg_rouge_l": f"{avg_rouge_l:.6f}",
        "gen_time_s": f"{gen_time_s:.2f}",
    }

    fieldnames = [
        "timestamp", "round", "group_id", "mode", "subset_size",
        "max_prompt_tokens", "max_new_tokens", "temperature", "top_p",
        "avg_em", "avg_rouge_l", "gen_time_s"
    ]

    lock = _file_locks[file_path]
    with lock:
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[Logger Error] Failed to write generation eval summary: {e}")


def save_generation_eval_samples(run_id, round_idx, group_id, mode, samples, max_items=50):
    """
    Saves examples (prompt/ref/pred) for auditing/debugging.
    Path: ../../results/{run_id}/metrics/gen_samples_round_{round}_group_{gid}_{mode}.json
    """
    dir_path = results_path(run_id, "metrics")
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(
        dir_path,
        f"gen_samples_round_{round_idx}_group_{group_id}_{mode}.json"
    )

    payload = {
        "run_id": run_id,
        "round": round_idx,
        "group_id": group_id,
        "mode": mode,
        "num_samples_saved": min(len(samples), max_items),
        "samples": samples[:max_items]
    }

    lock = _file_locks[file_path]
    with lock:
        try:
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[Logger Error] Failed to save generation samples: {e}")

# ============================================================
# Evaluation Helpers (LLM/Vision) - token/sample weighted sums
# ============================================================

def safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")

class StreamingEvalAccumulator:
    """
    Accumulator for streaming evaluation. Stores the sums (loss_sum, correct, n_tokens)
    and derives the averages at the end; n_tokens counts valid tokens for an LLM and
    samples for vision.
    """
    def __init__(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.n_tokens = 0

    def update(self, loss_sum: float, correct: int, n_tokens: int):
        if n_tokens <= 0:
            return
        self.loss_sum += float(loss_sum)
        self.correct += int(correct)
        self.n_tokens += int(n_tokens)



    def finalize(self, model_type: str):
        """
        Derives the final metrics from the accumulated sums, keeping
        avg_loss * num_tokens == loss_sum. For an LLM the accuracy is next-token
        prediction, not task accuracy, and avg_ppl is exp(avg_loss).
        """
        if self.n_tokens <= 0:
            base = {
                "loss_sum": 0.0,
                "correct": 0,
                "num_tokens": 0,
                "avg_loss": 0.0,
                "avg_ppl": 0.0,
            }
            if model_type == "llm":
                base["next_token_acc"] = 0.0
            else:
                base["avg_acc"] = 0.0
            return base

        avg_loss = self.loss_sum / self.n_tokens
        avg_acc = self.correct / self.n_tokens

        if model_type == "llm":
            avg_ppl = safe_exp(avg_loss)
            return {
                "loss_sum": float(self.loss_sum),
                "correct": int(self.correct),
                "num_tokens": int(self.n_tokens),
                "avg_loss": float(avg_loss),
                "avg_ppl": float(avg_ppl),
                "next_token_acc": float(avg_acc),
            }
        else:
            return {
                "loss_sum": float(self.loss_sum),
                "correct": int(self.correct),
                "num_tokens": int(self.n_tokens),
                "avg_loss": float(avg_loss),
                "avg_ppl": 0.0,
                "avg_acc": float(avg_acc),
            }

def _extract_logits(output):
    """
    Accepts either a HF output object (with .logits) or a raw tensor.
    Returns a tensor logits.
    """
    if hasattr(output, "logits"):
        return output.logits
    return output

def compute_eval_batch_metrics(model_type: str, output, labels, criterion, device=None):
    """
    Computes the per-batch sums (loss_sum, correct, n_tokens). The criterion must be
    CrossEntropyLoss with ignore_index=-100; for an LLM the logits and labels are
    shifted and the masked positions dropped.
    """
    logits = _extract_logits(output)
    if device is not None:
        logits = logits.to(device)
        labels = labels.to(device)

    if model_type == "llm":
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        del logits

        shift_labels = labels[..., 1:].contiguous()
        del labels

        # CE mean over valid tokens (ignore_index=-100)
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        preds = shift_logits.argmax(dim=-1)
        mask = (shift_labels != -100)

        n_tokens = int(mask.sum().item())
        if n_tokens > 0:
            correct = int((preds[mask] == shift_labels[mask]).sum().item())
            loss_sum = float(loss.item()) * n_tokens
        else:
            correct = 0
            loss_sum = 0.0

        return loss_sum, correct, n_tokens

    else:
        # Vision
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct = int((preds == labels).sum().item())
        n_tokens = int(labels.size(0))
        loss_sum = float(loss.item()) * n_tokens if n_tokens > 0 else 0.0
        return loss_sum, correct, n_tokens

# ============================================================
# Optional: generation metrics helpers (no extra deps)
# ============================================================

def normalize_text(s: str) -> str:
    import re
    import string
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s

def normalized_exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0

def _lcs_length(a: str, b: str) -> int:
    """
    Longest common subsequence length, token-level.
    Used for a lightweight ROUGE-L (F1).
    """
    a_tokens = a.split()
    b_tokens = b.split()
    n, m = len(a_tokens), len(b_tokens)
    if n == 0 or m == 0:
        return 0

    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def rouge_l_f1(pred: str, ref: str) -> float:
    """
    Lightweight ROUGE-L F1, token-level.
    """
    pred = pred.strip()
    ref = ref.strip()
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    prec = lcs / max(1, len(pred.split()))
    rec = lcs / max(1, len(ref.split()))
    if (prec + rec) == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def fixed_subset(indices, subset_size=200, seed=123):
    import random
    rng = random.Random(seed)
    idxs = list(indices)
    rng.shuffle(idxs)
    return idxs[:subset_size]

def malloc_trim_if_possible():
    """Attempts to return heap to the OS (Linux/glibc). Helps after large allocations."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

class WeightedAccumulator:
    """
    Memory-efficient accumulator for FedAvg.
    Progressively sums (weights * n_samples).
    Divides by total at finalization.
    Avoids keeping N model copies in RAM.
    """

    def __init__(self):
        self.acc = {}
        self.total_samples = 0
        self.non_tensor = {}

    def add(self, state_dict, n_samples):
        """Adds a model to the accumulator."""
        if n_samples <= 0: return

        self.total_samples += n_samples

        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                if k not in self.non_tensor:
                    self.non_tensor[k] = v
                continue

            t = v.detach().cpu()
            if t.is_floating_point():
                t = t.float()

            weighted = t.mul(n_samples)

            if k not in self.acc:
                self.acc[k] = weighted
            else:
                self.acc[k].add_(weighted)



    def finalize(self):
        """Computes the final average (divides by total)."""
        if self.total_samples == 0:
            return {}

        avg_sd = {}
        for k, v_sum in self.acc.items():
            avg_sd[k] = v_sum.div(self.total_samples)

        avg_sd.update(self.non_tensor)
        return avg_sd



