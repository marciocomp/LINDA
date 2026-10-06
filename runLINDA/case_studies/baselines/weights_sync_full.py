# @author: Marcio Lopes
"""
Weight synchronization for the *_full baselines.

The published SFL and HSFL baselines only train: they never exchange or average
weights, so a round of theirs has no synchronization cost, while LINDA collects,
aggregates and redistributes the model every round. The *_full variants add that
phase so the two costs can be compared side by side, and report it with the same
metric names LINDA uses (T_collect_weight, T_aggregation, T_distribution_weight).

Shards are packed exactly like LINDA's: {"client_id", "shard": {tier,start,end},
"weights": state_dict}, and the messages reuse LINDA's names, so the per-message
network records classify them as weights without any change.

The average is weighted by the number of samples each client contributed, over
shards that cover the same layer range, as in the layer-wise FedAvg of the
orchestrator.
"""
import time

import torch

REQUEST = "REQUEST_LOCAL_WEIGHTS"
RESPONSE = "LOCAL_WEIGHTS_RESPONSE"
RESPONSE_TAGGED = "LOCAL_WEIGHTS_RESPONSE_TAGGED"
UPDATE = "UPDATE_LOCAL_WEIGHTS"
UPDATE_TAGGED = "UPDATE_LOCAL_WEIGHTS_TAGGED"
DONE = "UPDATE_DONE"


def pack(model, client_id, tier, start, end, fp16=True, samples=0):
    """One shard of one participant, on host memory, ready to be sent."""
    sd = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        if fp16 and t.is_floating_point():
            t = t.half()
        sd[k] = t
    return {"client_id": client_id, "samples": int(samples),
            "shard": {"tier": tier, "start": int(start), "end": int(end)},
            "weights": sd}


def samples_map(packs):
    """participant -> samples, taken from the packs themselves."""
    out = {}
    for p in packs:
        n = int(p.get("samples", 0))
        if n > 0:
            out[p.get("client_id")] = max(out.get(p.get("client_id"), 0), n)
    return out


def signature(pack_):
    sh = pack_.get("shard", {})
    return str(sh.get("tier")), int(sh.get("start", -1)), int(sh.get("end", -1))


def fed_avg(packs, samples_by_client):
    """
    Weighted FedAvg per shard signature.

    packs: list of packs (several clients, several shards).
    samples_by_client: client_id -> number of samples trained.
    Returns one averaged pack per signature, keys untouched.
    """
    acc, wsum, first_non_float, template = {}, {}, {}, {}

    for p in packs:
        sig = signature(p)
        n = float(samples_by_client.get(p.get("client_id"), 0))
        if n <= 0:
            continue
        template.setdefault(sig, p["shard"])
        for k, v in p["weights"].items():
            if not torch.is_tensor(v) or not v.is_floating_point():
                first_non_float.setdefault((sig, k), v)
                continue
            key = (sig, k)
            if key not in acc:
                acc[key] = torch.zeros(v.shape, dtype=torch.float32)
                wsum[key] = 0.0
            acc[key].add_(v.to(torch.float32), alpha=n)
            wsum[key] += n

    averaged = {}
    for (sig, k), tensor in acc.items():
        denom = wsum[(sig, k)]
        averaged.setdefault(sig, {})[k] = tensor.div_(denom) if denom > 0 else tensor
    for (sig, k), value in first_non_float.items():
        averaged.setdefault(sig, {})[k] = value

    return [{"client_id": "global", "shard": template[sig], "weights": sd}
            for sig, sd in averaged.items()]


def match(packs, tier, start, end):
    """The averaged pack covering one layer range, or None."""
    for p in packs:
        if signature(p) == (str(tier), int(start), int(end)):
            return p
    return None


def first_of_tier(packs, tier):
    """The averaged pack of a tier, whatever its layer range."""
    for p in packs:
        if str(p.get("shard", {}).get("tier")) == str(tier):
            return p
    return None


def upstream_of(packs, start_layer):
    """
    Averaged shards that still have to travel upstream from a node holding blocks
    from start_layer on: only the ones covering earlier layers. A node applies its
    own shard and never forwards it again, and the shards of the nodes below it
    were already applied on the way, so passing the whole list along the chain only
    moves bytes nobody reads.
    """
    return [p for p in packs if int(p.get("shard", {}).get("start", 0)) < int(start_layer)]


def apply_to(model, pack_):
    """Loads an averaged shard into a model; dtype is converted by load_state_dict."""
    if pack_ is None:
        return False
    model.load_state_dict(pack_["weights"], strict=False)
    return True


class PhaseTimer:
    """Times the synchronization phases and logs them with LINDA's metric names."""

    def __init__(self, helpers, node_id, run_id, seed, round_idx=0):
        self.helpers, self.node_id, self.run_id = helpers, node_id, run_id
        self.seed, self.round_idx = seed, round_idx
        self.start = time.perf_counter()
        self._mark = self.start

    def phase(self, metric_name, client_id=None):
        now = time.perf_counter()
        self._log(metric_name, now - self._mark, client_id)
        self._mark = now

    def total(self, metric_name="T_sync_phase"):
        self._log(metric_name, time.perf_counter() - self.start, None)

    def _log(self, metric_name, duration, client_id):
        self.helpers.log_performance_metric(
            node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
            metric_name=metric_name, duration=duration, round_idx=self.round_idx,
            batch_idx=-1, req_id=None, delay=None)
