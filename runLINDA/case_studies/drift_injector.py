#!/usr/bin/env python3
"""
Takes memory away from a GPU while a run is using it.

Experiment 3 needs the resource state to change during training. The honest way
to produce that is a second process competing for the same card, which is what
happens on a shared machine: the memory really stops being available, a node
sized for the old budget really does fail, and the orchestrator learns about it
the same way it would in production -- from what each node reports between
rounds. Nothing is edited in a file and no code is told to pretend.

It is topology-agnostic: it knows about a device, not about sites, tiers or
chains. Run one instance per card you want to squeeze, on the host that holds it.

    # take 8 GiB of cuda:1 at 14:30, and hold until stopped
    python drift_injector.py --device cuda:1 --gib 8 --start-at 14:30

    # before a run starts, "leave this much free" is the reproducible form
    python drift_injector.py --device cuda:1 --leave-free 14 --trigger-file /tmp/drift

    # take memory AND keep the card busy a tenth of the time
    python drift_injector.py --device cuda:1 --gib 7 --gpu-load 10

    # see what it would do, without touching anything
    python drift_injector.py --device cuda:1 --leave-free 14 --dry-run

What the node on the other side sees: its own reading is `free + what that node's
own process reserved`, so memory held by a stranger shows up as a smaller budget,
which is exactly what the allocation has to be computed against. Taking X GiB
therefore lowers that node's budget by X, whatever its shard happens to occupy --
which is why `--gib` is the form to use once training is under way, and
`--leave-free` the one to use against an idle card.

`--gpu-load` competes for the card's arithmetic as well, by running matmuls for a
fraction of the time. That reaches the other resource the allocation reads: the
throughput each node measures on its own device, which is what decides the share a
faster node is offered. Be aware of how little a small load is likely to change
that reading, though: the victim's probe lasts tens of milliseconds and takes the
median of three, so a load that occupies the card a tenth of the time will usually
miss it altogether. Ten percent is a realistic neighbour and will show in
`nvidia-smi` and in the per-batch times; moving the allocation through the
throughput term needs a far higher duty cycle.

SCHEDULED MODE (`--schedule`)
-----------------------------
A campaign that compares re-allocation on against off needs every seed to suffer
the SAME drift at the SAME moments. Firing by wall clock cannot do that: rounds
last around 2,400 s but they vary, and an injection that lands in the wrong phase
changes which round the re-allocation reaches. So the injector reads the run's own
progress instead.

    # the same command on every host that holds one of the target cards
    python drift_injector.py --schedule --run-id 202609281530 --gib 7 5 15

The three amounts are positional, one per step of `SCHEDULE` below: the first
squeezes `s2_t2_1` during round 0, the second releases `s2_t2_1` and squeezes
`s2_t2_2` during round 1, the third squeezes `s1_t2_1` during round 2. Each step
acts DURING round r so that the decision taken at the end of round r takes effect
in round r+1.

WHY DURING ROUND r AND NOT BETWEEN r AND r+1. The orchestrator decides inside
round r, after its evaluation: T_train_phase -> weight collection -> aggregation
-> distribution -> evaluation -> DISCOVERY + DECISION + APPLY -> T_global_cycle.
Firing on the round's real end is already too late: the decision has happened, the
drift would only be seen at the end of r+1, and the last step would land on the
final round, where no placement is computed at all. Firing when the training phase
of round r ends leaves about 815 s of margin before the discovery, measured over
the five baseline runs, which is room enough for a three-minute poll.

Each host runs the same command and picks out the steps whose target card is its
own, by matching the node's IP in the topology against its own interfaces. A `0`
amount skips that step's take but not its release.

COMPLETING A TAKE. That margin has a catch at its near end: the marker is written
while the victim is still at its TRAINING peak, a couple of GiB above the
footprint it settles at once `cluster_manager` frees its cache, a second later. A
take that lands there gets less than it asked for and says so only on stdout --
which is how the five seeds of the September campaign each removed 6.0 GiB where 7
were asked, identically, deterministically, and invisibly in the saved CSVs. So a
short take is retried every `--retry-seconds` until it is whole or
`--take-timeout` passes, and a take that stays short ABORTS the seed for the same
reason a late marker does: a seed that lost a different amount from the others is
mislabelled, which is worse than a seed that is missing. `--allow-partial` keeps
the old behaviour. What the experiment reports either way is the measured
`held_by_others_gib` of `profiles_*.csv`, not the amount on the command line.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import signal
import socket
import sys
import threading
import time

import psutil
import torch

GIB = 1024 ** 3
CHUNK_GIB = 0.5          # allocated in pieces, so fragmentation cannot refuse the whole
TAKE_TOLERANCE = 0.05    # GiB; a take this close to the target IS the target
LOAD_N = 2048            # square fp32 matmul: three of these are about 48 MiB
LOAD_TARGET_BURST_S = 0.03   # a burst long enough that the sleep between them is sane

HERE = os.path.dirname(os.path.abspath(__file__))

# One step per amount given to `--gib`, in that order: the round during which to
# act, the node whose card is squeezed, and the node to release first (None when
# nothing is released). This is the drift the Experiment 3 campaign injects, and it
# is a constant rather than a flag so that every seed gets the same shape and only
# the amounts are chosen at the command line.
SCHEDULE = (
    (0, "s2_t2_1", None),
    (1, "s2_t2_2", "s2_t2_1"),
    (2, "s1_t2_1", None),
)

# The metric whose appearance marks the end of a round's training phase, and the
# node that writes it. See the module docstring for why this marker and not the
# round's real end.
DEFAULT_WATCH = "s1_t3_1:T_train_phase"


class GpuLoad(threading.Thread):
    """Occupies the card for a fraction of the time, in bursts.

    A GPU is either running a kernel or it is not, so "a tenth of the card" is a
    tenth of the time rather than a tenth of the silicon: the loop computes for a
    while, then sleeps for nine times as long. That is also what `nvidia-smi`
    reports as utilization, and it is what a neighbouring job actually looks like.

    The burst is calibrated once, so that it lasts long enough for the sleep to be
    accurate and short enough not to stall the victim for a visible time. How much
    of the card it really took is measured rather than assumed, and printed on the
    way out.
    """

    def __init__(self, device, percent):
        super().__init__(daemon=True)
        self.device = device
        self.percent = max(0.0, min(100.0, float(percent)))
        self.stop = threading.Event()
        self.busy_s = 0.0
        self.elapsed_s = 0.0
        self.scratch_gib = 0.0

    def run(self):
        if self.percent <= 0:
            return
        a = torch.randn((LOAD_N, LOAD_N), device=self.device, dtype=torch.float32)
        b = torch.randn((LOAD_N, LOAD_N), device=self.device, dtype=torch.float32)
        c = a @ b
        torch.cuda.synchronize(self.device)
        self.scratch_gib = 3 * (LOAD_N ** 2) * 4 / GIB

        # One matmul, timed, so the burst can be sized in iterations.
        started = time.perf_counter()
        for _ in range(5):
            c = a @ b
        torch.cuda.synchronize(self.device)
        per_iter = max((time.perf_counter() - started) / 5, 1e-6)
        iters = max(1, int(LOAD_TARGET_BURST_S / per_iter))

        began = time.perf_counter()
        while not self.stop.is_set():
            t0 = time.perf_counter()
            for _ in range(iters):
                c = a @ b
            torch.cuda.synchronize(self.device)
            burst = time.perf_counter() - t0
            self.busy_s += burst
            if self.percent < 100.0:
                self.stop.wait(burst * (100.0 - self.percent) / self.percent)
        self.elapsed_s = time.perf_counter() - began

        del a, b, c
        torch.cuda.empty_cache()

    def report(self):
        if self.percent <= 0 or self.elapsed_s <= 0:
            return
        print("[drift] the load held the card %.1f%% of the time over %.0f s "
              "(asked for %.0f%%)"
              % (100.0 * self.busy_s / self.elapsed_s, self.elapsed_s, self.percent),
              flush=True)


class Holdings:
    """The memory this process is taking, per card.

    Scheduled mode moves between cards over the run -- it squeezes one 4090, gives
    it back and squeezes the other -- so what is held cannot be a single list. Each
    card's tensors are kept apart and can be released independently.

    Allocated in pieces rather than as one block: a single allocation of many GiB
    can be refused by fragmentation on a card that is already in use, which is
    precisely the card this is meant to run against.
    """

    def __init__(self):
        self.held = {}

    def take(self, device, amount):
        key = str(device)
        chunk_elements = int(CHUNK_GIB * GIB / 4)
        pieces, taken = self.held.setdefault(key, []), 0.0
        try:
            while taken + CHUNK_GIB <= amount:
                pieces.append(torch.empty(chunk_elements, dtype=torch.float32,
                                          device=device))
                taken += CHUNK_GIB
            rest = amount - taken
            if rest > 0.01:
                pieces.append(torch.empty(int(rest * GIB / 4), dtype=torch.float32,
                                          device=device))
                taken += rest
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            print("[drift] could only take %.2f GiB of %s: %s" % (taken, key, exc),
                  flush=True)
        return taken

    def release(self, device):
        key = str(device)
        pieces = self.held.pop(key, [])
        if not pieces:
            return False
        del pieces
        torch.cuda.empty_cache()
        return True

    def release_all(self):
        for key in list(self.held):
            self.release(key)


class StopRequest:
    """Ctrl+C asks for a clean stop; a second one stops waiting for it.

    The handler cannot do the releasing itself -- it runs between bytecodes, and
    freeing tensors there would race with whatever allocated them -- so it raises a
    flag the loops watch. What makes that slow is `time.sleep`: since PEP 475 it
    runs the handler at once and then sleeps out the REST of its interval, so one
    long sleep keeps the process alive for up to `--poll-seconds` after the key was
    pressed. `nap` therefore sleeps in one-second slices.

    A second signal calls `os._exit`, skipping the release and the report. That is
    safe: the driver reclaims a dead process's memory whether or not the tensors
    were freed politely. The clean path exists for the log, not for the card.
    """

    def __init__(self):
        self.requested = False

    def install(self):
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        return self

    def _on_signal(self, signum, frame):
        if self.requested:
            print("[drift] second signal: leaving now, without releasing politely. "
                  "The driver reclaims the memory when this process dies.", flush=True)
            os._exit(130)
        self.requested = True
        print("[drift] stop requested: releasing what is held, then exiting. "
              "Press again to leave immediately.", flush=True)

    def nap(self, seconds):
        """Sleeps up to `seconds`, returning early if a stop was requested."""
        end = time.monotonic() + max(0.0, seconds)
        while not self.requested:
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(1.0, left))


def resolve_target(topology, node_id):
    """The (ip, device) of a Tier 2 node, from the topology.

    A six-line walk of the same dict `helpers.resolve_node_ip` walks, done here so
    that this tool keeps `torch` and `psutil` as its only project-external imports:
    it has to run on whichever host holds a card, and it should not fail to squeeze
    a GPU because something unrelated in `utils.helpers` could not be imported.
    """
    for site in topology["sites"].values():
        for node in (site.get("tier2") or {}).values():
            if node["id"] == node_id:
                return node.get("ip"), node.get("device")
    return None, None


def local_ipv4s(override=None):
    """Every IPv4 address this host answers on, so a step can tell if it is ours."""
    if override:
        return {override}
    found = set()
    for addrs in psutil.net_if_addrs().values():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                found.add(addr.address)
    return found


def metrics_file(results_dir, run_id, node_id):
    """The performance-metrics CSV of one node, whose seed is not known here."""
    pattern = os.path.join(results_dir, run_id, "performance_metrics",
                           "%s_seed_*_date_%s_metrics.csv" % (node_id, run_id))
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def find_round_row(path, metric, round_idx):
    """The epoch timestamp of `metric` for `round_idx`, or None if not written yet.

    The row carries the `time.time()` of whoever wrote it, which is what lets the
    caller tell how long ago the round actually ended -- on a host that sees the
    file through a synchronised folder rather than locally, that gap is the risk.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("metric") != metric:
                    continue
                try:
                    if int(row.get("round", -1)) == int(round_idx):
                        return float(row["timestamp"])
                except (TypeError, ValueError):
                    continue
    except OSError as exc:
        print("[drift] could not read %s: %s" % (path, exc), flush=True)
    return None


def plan_actions(args, topology, local):
    """The take/release actions of `SCHEDULE`, with the ones for this host marked.

    Release and take are separate actions because in general they can sit on
    different hosts; a host executes only what is its own.
    """
    actions = []
    for index, (round_idx, target, release_first) in enumerate(SCHEDULE):
        amount = float(args.gib[index])
        if release_first:
            ip, device = resolve_target(topology, release_first)
            actions.append({"round": round_idx, "kind": "release", "node": release_first,
                            "ip": ip, "device": device, "gib": 0.0,
                            "mine": ip in local})
        ip, device = resolve_target(topology, target)
        actions.append({"round": round_idx, "kind": "take", "node": target,
                        "ip": ip, "device": device, "gib": amount,
                        "mine": ip in local})
    return actions


def describe_actions(actions):
    for a in actions:
        print("   round %d  %-7s %-9s %-7s %-14s %s%s"
              % (a["round"], a["kind"], a["node"], a["device"] or "?", a["ip"] or "?",
                 ("%.2f GiB" % a["gib"]) if a["kind"] == "take" else "-",
                 "" if a["mine"] else "   (another host)"), flush=True)


def take_in_full(holdings, device, amount, stop, args, label):
    """Take `amount` GiB, completing the take instead of accepting what first fits.

    A node's training peak is not the footprint it settles at: `cluster_manager`
    frees its cache right after joining the round's threads, so for a moment the
    card holds a couple of GiB more than it will a second later. The marker this
    schedule fires on is written inside that moment, which is how the five seeds of
    the September campaign all recorded 6.0 GiB where 7 were asked -- deterministic,
    not a race, and invisible unless somebody reads the profiles afterwards. What
    the experiment means is the memory the victim LOSES, so the take is retried
    until it is whole.

    Returns (held, attempts, complete).
    """
    deadline = time.time() + args.take_timeout
    held, attempts = 0.0, 0
    while True:
        attempts += 1
        held += holdings.take(device, amount - held)
        if held + TAKE_TOLERANCE >= amount:
            return held, attempts, True
        if stop.requested or time.time() >= deadline:
            return held, attempts, False
        print("[drift] %s: holding %.2f of %.2f GiB after attempt %d; the card is "
              "still busy, retrying in %.0f s"
              % (label, held, amount, attempts, args.retry_seconds), flush=True)
        stop.nap(args.retry_seconds)


def run_schedule(args):
    """Squeezes the cards of this host at the rounds `SCHEDULE` names."""
    if not os.path.exists(args.topology):
        sys.exit("[drift] topology not found: %s" % args.topology)
    topology = json.load(open(args.topology))

    if len(args.gib) != len(SCHEDULE):
        sys.exit("[drift] --schedule needs %d amounts for --gib, one per step; got %d"
                 % (len(SCHEDULE), len(args.gib)))

    local = local_ipv4s(args.host_ip)
    actions = plan_actions(args, topology, local)
    mine = [a for a in actions if a["mine"]]

    watch_node, _, watch_metric = args.watch.partition(":")
    path = metrics_file(args.results_dir, args.run_id, watch_node)

    print("[drift] run %s | watching %s of %s" % (args.run_id, watch_metric, watch_node),
          flush=True)
    print("[drift] this host answers on %s" % ", ".join(sorted(local)), flush=True)
    print("[drift] full schedule:", flush=True)
    describe_actions(actions)
    print("[drift] %d action(s) are this host's" % len(mine), flush=True)
    print("[drift] metrics file: %s" % (path or "not written yet"), flush=True)

    if args.dry_run:
        # A replay, so the monitor can be checked against a run that already
        # finished without touching a single byte of memory.
        for round_idx in range(args.rounds):
            ts = find_round_row(path, watch_metric, round_idx)
            print("   round %d  %s" % (round_idx,
                  "not found" if ts is None else
                  "%s (%.0f s ago)" % (datetime.datetime.fromtimestamp(ts)
                                       .isoformat(timespec="seconds"),
                                       time.time() - ts)), flush=True)
        print("[drift] dry run: nothing was taken.", flush=True)
        return 0

    if not mine:
        print("[drift] nothing to do on this host; exiting.", flush=True)
        return 0

    for device in sorted({a["device"] for a in mine if a["device"]}):
        if not str(device).startswith("cuda"):
            sys.exit("[drift] %s is not a CUDA device" % device)
    if not torch.cuda.is_available():
        sys.exit("[drift] torch reports no CUDA device on this host")
    torch.cuda.init()

    holdings = Holdings()
    stop = StopRequest().install()

    def wait_for(round_idx):
        """Blocks until the marker of `round_idx` appears; returns its lag, or None."""
        while not stop.requested:
            ts = find_round_row(path, watch_metric, round_idx)
            if ts is not None:
                return time.time() - ts
            stop.nap(args.poll_seconds)
        return None

    rounds_with_work = sorted({a["round"] for a in mine})
    for round_idx in rounds_with_work:
        print("[drift] waiting for round %d of %s" % (round_idx, args.run_id), flush=True)
        lag = wait_for(round_idx)
        if lag is None:
            break

        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        print("[drift] %s: round %d's training phase ended %.0f s ago"
              % (stamp, round_idx, lag), flush=True)
        if lag > args.max_lag and not args.allow_late:
            # Acting now would land after the orchestrator's discovery, so the drift
            # would reach a different round than the schedule says. A seed recorded
            # that way is mislabelled, which is worse than a seed that is missing.
            print("[drift] ABORTING: the marker arrived %.0f s late, over the %.0f s "
                  "allowed, so this step would miss its decision. Discard this seed. "
                  "If this host reads the file through a synchronised folder, re-run "
                  "with --watch pointing at a node that writes locally."
                  % (lag, args.max_lag), flush=True)
            holdings.release_all()
            return 2

        for action in [a for a in mine if a["round"] == round_idx]:
            device = torch.device(action["device"])
            report(device, "before")
            if action["kind"] == "release":
                freed = holdings.release(device)
                print("[drift] released %s (%s)%s"
                      % (action["device"], action["node"],
                         "" if freed else " -- nothing was held here"), flush=True)
            elif action["gib"] <= 0:
                print("[drift] %s (%s): 0 GiB asked for, skipped."
                      % (action["device"], action["node"]), flush=True)
            else:
                label = "%s (%s)" % (action["device"], action["node"])
                taken, attempts, complete = take_in_full(
                    holdings, device, action["gib"], stop, args, label)
                print("[drift] holding %.2f GiB of %s%s"
                      % (taken, label,
                         "" if attempts == 1 else " after %d attempts" % attempts),
                      flush=True)
                if not complete and not args.allow_partial:
                    # Same reasoning as the lag guard: a seed that lost a different
                    # amount from the others is mislabelled, which is worse than a
                    # seed that is missing.
                    print("[drift] ABORTING: could only take %.2f of the %.2f GiB "
                          "asked for on %s within %.0f s. Discard this seed, or "
                          "re-run with --allow-partial to accept what fits."
                          % (taken, action["gib"], label, args.take_timeout),
                          flush=True)
                    holdings.release_all()
                    return 3
            report(device, "after")

    if not stop.requested:
        last = args.rounds - 1
        print("[drift] schedule done; holding until round %d ends" % last, flush=True)
        lag = wait_for(last)
        if lag is not None and args.tail_seconds > 0:
            print("[drift] round %d's training ended; holding %.0f s more while the "
                  "run finishes" % (last, args.tail_seconds), flush=True)
            stop.nap(args.tail_seconds)

    holdings.release_all()
    for device in sorted({a["device"] for a in mine if a["device"]}):
        report(torch.device(device), "released")
    print("[drift] done.", flush=True)
    return 0


def parse_moment(value):
    """An epoch timestamp, or a wall-clock HH:MM[:SS] later today."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    parts = [int(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.append(0)
    now = datetime.datetime.now()
    target = now.replace(hour=parts[0], minute=parts[1], second=parts[2], microsecond=0)
    if target < now:
        target += datetime.timedelta(days=1)
    return target.timestamp()


def read(device):
    free, total = torch.cuda.mem_get_info(device)
    return free / GIB, total / GIB


def report(device, tag):
    free, total = read(device)
    print("[drift] %-9s %s  free %.2f of %.2f GiB" % (tag, device, free, total), flush=True)
    return free


def wanted(args, free_now):
    """How much to take, from whichever way the operator chose to say it."""
    if args.gib is not None:
        return args.gib
    # `--leave-free` names the state to arrive at rather than the amount to remove,
    # so the same command has the same effect on a card whose baseline moved since
    # the last run. IT REFERS TO THE MEMORY THE CARD REPORTS FREE RIGHT NOW, which
    # is not the budget the node on the other side reports: that node adds back its
    # own allocator pool, because a re-planned shard could reuse it. Before the run
    # builds its shards the two coincide; once training has started, free is smaller
    # than the budget by the size of the shard. To move the budget by a given amount
    # mid-run, say `--gib` -- taking X GiB lowers the node's budget by exactly X.
    return max(0.0, free_now - args.leave_free)


def main():
    ap = argparse.ArgumentParser(description="Competes for a GPU, to make a run's "
                                             "resource state change.")
    ap.add_argument("--device", default="cuda:0", help="the card to compete for")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--gib", type=float, nargs="+",
                      help="how much to take: one amount, or one per step of the "
                           "schedule when --schedule is given")
    size.add_argument("--leave-free", type=float,
                      help="take whatever leaves this much free")
    ap.add_argument("--start-at", default=None,
                    help="epoch seconds or HH:MM[:SS]; default is immediately")
    ap.add_argument("--trigger-file", default=None,
                    help="wait until this path exists, then take the memory")
    ap.add_argument("--hold-seconds", type=float, default=None,
                    help="release after this long; default is to hold until stopped")
    ap.add_argument("--gpu-load", type=float, default=0.0,
                    help="also compete for the card's arithmetic, this percent of "
                         "the time (0 disables it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be taken and stop")

    sch = ap.add_argument_group("scheduled mode",
                               "drive the injections from a run's own progress")
    sch.add_argument("--schedule", action="store_true",
                     help="act at the rounds SCHEDULE names, instead of once")
    sch.add_argument("--run-id", default=None,
                     help="the run to follow; 'latest' takes the newest one")
    sch.add_argument("--rounds", type=int, default=4,
                     help="how many rounds the run has, to know when it ends")
    sch.add_argument("--poll-seconds", type=float, default=180.0,
                     help="how often to look at the run's metrics")
    sch.add_argument("--results-dir", default=os.path.join(HERE, "results"),
                     help="where the runs write; default is next to this script")
    sch.add_argument("--topology", default=os.path.join(HERE, "cds",
                                                        "network_config_cds.json"),
                     help="topology that maps a node to its host and card")
    sch.add_argument("--watch", default=DEFAULT_WATCH,
                     help="node:metric whose appearance marks a round; default "
                          + DEFAULT_WATCH)
    sch.add_argument("--max-lag", type=float, default=400.0,
                     help="abort if a marker is seen this long after it was written, "
                          "since the step would then miss its decision")
    sch.add_argument("--allow-late", action="store_true",
                     help="act anyway when the marker is late")
    sch.add_argument("--take-timeout", type=float, default=240.0,
                     help="how long to keep completing a take that the card refused "
                          "in full; it comes out of the same margin as --max-lag")
    sch.add_argument("--retry-seconds", type=float, default=10.0,
                     help="how long to wait between attempts at completing a take")
    sch.add_argument("--allow-partial", action="store_true",
                     help="accept a take that stayed short instead of aborting")
    sch.add_argument("--tail-seconds", type=float, default=120.0,
                     help="how long to keep holding after the last round's training")
    sch.add_argument("--host-ip", default=None,
                     help="pretend this host answers on this address")
    args = ap.parse_args()

    if args.schedule:
        if args.leave_free is not None:
            sys.exit("[drift] --schedule takes --gib, not --leave-free: the amounts "
                     "have to mean the same thing on every seed.")
        if not args.gib:
            sys.exit("[drift] --schedule needs --gib with one amount per step (%d)"
                     % len(SCHEDULE))
        if not args.run_id:
            sys.exit("[drift] --schedule needs --run-id (or --run-id latest)")
        if args.run_id == "latest":
            runs = sorted(d for d in glob.glob(os.path.join(args.results_dir, "2*"))
                          if os.path.isdir(d))
            if not runs:
                sys.exit("[drift] no run found under %s" % args.results_dir)
            args.run_id = os.path.basename(runs[-1])
            print("[drift] latest run is %s" % args.run_id, flush=True)
        sys.exit(run_schedule(args))

    if args.gib is None and args.leave_free is None:
        sys.exit("[drift] give --gib or --leave-free, or --schedule")
    if args.gib is not None and len(args.gib) != 1:
        sys.exit("[drift] without --schedule, --gib takes a single amount")
    if args.gib is not None:
        args.gib = args.gib[0]

    device = torch.device(args.device)
    if device.type != "cuda":
        sys.exit("[drift] --device has to be a CUDA device")
    if not torch.cuda.is_available():
        sys.exit("[drift] torch reports no CUDA device on this host")

    torch.cuda.init()
    free_now = report(device, "before")
    amount = wanted(args, free_now)

    if amount <= 0 and args.gpu_load <= 0:
        print("[drift] nothing to take: no memory asked for and no load asked for.",
              flush=True)
        return
    if amount < 0:
        amount = 0.0
    if amount > free_now:
        print("[drift] asked for %.2f GiB but only %.2f are free; taking what there is."
              % (amount, free_now), flush=True)
        amount = free_now

    if args.dry_run:
        print("[drift] would take %.2f GiB of %s, leaving %.2f free, and hold the "
              "card %.0f%% of the time."
              % (amount, device, free_now - amount, args.gpu_load), flush=True)
        return

    # Installed before the waiting starts, so Ctrl+C behaves the same whether it
    # arrives while waiting for the moment, for the trigger file, or while holding.
    stop = StopRequest().install()

    moment = parse_moment(args.start_at)
    if moment is not None:
        wait = moment - time.time()
        if wait > 0:
            print("[drift] waiting %.1f s, until %s"
                  % (wait, datetime.datetime.fromtimestamp(moment)), flush=True)
            stop.nap(wait)

    if args.trigger_file and not stop.requested:
        print("[drift] waiting for %s" % args.trigger_file, flush=True)
        while not os.path.exists(args.trigger_file) and not stop.requested:
            stop.nap(0.5)

    if stop.requested:
        print("[drift] stopped before taking anything.", flush=True)
        return

    # The load starts first, and what its scratch occupies counts towards the
    # amount asked for: `--gib X` means the victim loses X GiB, not X plus whatever
    # this process needed to keep itself busy.
    load = GpuLoad(device, args.gpu_load)
    if args.gpu_load > 0:
        load.start()
        while load.scratch_gib == 0.0 and load.is_alive():
            time.sleep(0.05)
        amount = max(0.0, amount - load.scratch_gib)
        print("[drift] load running; its scratch holds %.3f GiB, leaving %.2f to hold"
              % (load.scratch_gib, amount), flush=True)

    holdings = Holdings()
    taken = holdings.take(device, amount)

    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    print("[drift] %s: holding %.2f GiB of %s" % (stamp, taken, device), flush=True)
    report(device, "after")

    deadline = time.time() + args.hold_seconds if args.hold_seconds else None
    while not stop.requested and (deadline is None or time.time() < deadline):
        stop.nap(0.5)

    load.stop.set()
    if load.is_alive():
        load.join(10)
    load.report()

    holdings.release_all()
    report(device, "released")


if __name__ == "__main__":
    main()
