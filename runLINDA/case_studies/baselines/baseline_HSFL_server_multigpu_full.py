# @author: Marcio Lopes
# Same training path as baseline_HSFL_server_multigpu.py, plus the weight synchronization phase the
# published baseline does not have. The shards travel along the chain the
# architecture already uses (clients -> stages -> orchestrator), are averaged with
# a sample-weighted FedAvg and come back the same way, so the cost of a federated
# round can be compared with LINDA's. The training loop is untouched, so
# T_total_train_time stays comparable with the original baseline.
#
# Static HSFL Tier 2 stage for the MULTI-GPU chain variant.
#
# One process per chain GPU, chained over TCP. This mirrors how LINDA's Tier 2
# Compute Nodes actually run -- separate processes exchanging pickled tensors
# over sockets even when co-located on the same host -- so that both systems pay
# the same per-hop serialization cost and the comparison stays fair.
#
# Each stage is the same component: it accepts group_size persistent upstream
# sockets, holds one shard plus optimizer per slot, and forwards to a next hop
# that is either the following stage or the Tier 3 orchestrator. Stage 0
# additionally accepts the Tier 1 clients and owns the control plane, exactly as
# the single-GPU baseline_HSFL_server_t2.py does today.
#
# Layer split for the minimal_tail variant (Tier 1 keeps block 0):
#   stage0 cuda:5  blocks  1-7    stage1 cuda:4  blocks  7-13
#   stage2 cuda:3  blocks 13-17   orch   cuda:0  blocks 17-18 + LM head
#
# Startup order is listen-before-connect: orch, then stage2, stage1, stage0,
# then the clients.
#
# Usage (from runLINDA/case_studies/cds/baselines/), see README of the plan:
#   python baseline_HSFL_server_multigpu.py --stage_id 0 --is_first \
#       --device cuda:5 --split_point_start 1 --split_point_end 7 \
#       --listen_port 9100 --next_port 9110

import torch
import torch.optim as optim
import threading
import time
import argparse
import os
import sys
from pathlib import Path
import gc
import socket
import tqdm

sys.path.append('../../../../..')
sys.path.append(str(Path(__file__).resolve().parents[5]))   # repo root, whatever the cwd

from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers
import weights_sync_full as wsync


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline HSFL Tier 2 - one chain stage")
    # chain position
    parser.add_argument("--stage_id", type=int, required=True)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--is_first", action="store_true",
                        help="accepts Tier 1 clients and owns the control plane")
    parser.add_argument("--is_last", action="store_true",
                        help="next hop is the Tier 3 orchestrator data port")

    # wiring
    parser.add_argument("--listen_ip", type=str, default="130.92.70.8")
    parser.add_argument("--listen_port", type=int, required=True)
    parser.add_argument("--next_ip", type=str, default="130.92.65.51",
                        help="proxy of 130.92.70.8: the next stage is on this same host, "
                             "so the hop has to cross the network instead of loopback")
    parser.add_argument("--next_port", type=int, required=True)
    parser.add_argument("--ip_orch", type=str, default="130.92.65.51",
                        help="proxy of 130.92.70.8, where the orchestrator runs")
    parser.add_argument("--port_orch_control", type=int, default=10101)

    # model shard
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--model_type", type=str, default="llm")
    parser.add_argument("--split_point_start", type=int, required=True)
    parser.add_argument("--split_point_end", type=int, required=True)

    # training
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--total_clients", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="only used to weight the FedAvg by samples")
    parser.add_argument("--clients_by_cycle", action="store_true",
                        help="first stage only: one client process per slot AND cycle, so the "
                             "clients of a cycle can sit on the same host and contend for it, "
                             "as the Tier 1 agents of LINDA do")
    return parser.parse_args()


class HSFLChainStage(CommunicationModule):
    def __init__(self, args):
        super().__init__(f"server_stage{args.stage_id}", args.listen_ip)
        self.args = args

        self.stage_id = int(args.stage_id)
        self.num_stages = int(args.num_stages)
        self.is_first = bool(args.is_first)
        self.is_last = bool(args.is_last)

        self.ip = args.listen_ip
        self.port = args.listen_port
        self.next_ip = args.next_ip
        self.next_port = args.next_port

        self.ip_orch = args.ip_orch
        self.port_orch_control = args.port_orch_control

        self.seed = args.seed
        self.lr = args.lr

        self.model_name = args.model_name
        self.model_type = args.model_type
        self.start_layer = args.split_point_start
        self.end_layer = args.split_point_end

        self.total_clients = int(args.total_clients)
        self.group_size = int(args.group_size)
        self.samples_by_slot = {}     # everything each slot trained, for the FedAvg weight
        self.num_cycles = (self.total_clients + self.group_size - 1) // self.group_size

        self.device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(int(args.device.split(":")[1]))

        self.run_id = None
        self.control_sock = None            # only on stage 0

        self.up_sock_by_slot = {}           # slot -> socket to the previous hop
        self.up_sock_by_cycle = {}          # (cycle, slot) -> socket, with --clients_by_cycle
        self.clients_by_cycle = bool(getattr(args, "clients_by_cycle", False))
        self.down_sock_by_slot = {}         # slot -> socket to the next hop

        self.slot_to_client = {}
        self.slot_to_iters = {}

        self.models_by_slot = {}
        self.opts_by_slot = {}

        self.clip_max_norm = float(args.clip_max_norm)
        self.initial_snapshot = helpers.get_local_resource_snapshot(self.device)

        logger.info(f"[stage{self.stage_id}] blocks {self.start_layer}-{self.end_layer} "
                    f"on {self.device} | first={self.is_first} last={self.is_last}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def connect_control_to_orch(self):
        """Stage 0 only: fetch the run id and keep the cycle-barrier socket."""
        self.sock_client.connect((self.ip_orch, self.port_orch_control))
        self.control_sock = self.sock_client
        self.send_msg(self.control_sock, ["HELLO_TIER2"])
        msg = self.recv_msg(self.control_sock, "RUN_ID")
        self.run_id = msg[1]
        self.enable_net_metrics(self.run_id)
        logger.info(f"[stage{self.stage_id}] control plane up, run_id={self.run_id}")

    def accept_upstream(self):
        """
        Accept group_size persistent sockets from the previous hop.

        A handshake carrying no run_id means the peer is a Tier 1 client, which
        blocks waiting for RUN_ID; a handshake carrying one means the peer is the
        preceding chain stage, and the id propagates from it.
        """
        self.sock_server.bind((self.ip, self.port))
        self.sock_server.listen(20)
        expected = self.group_size * self.num_cycles if self.clients_by_cycle else self.group_size
        logger.info(f"[stage{self.stage_id}] listening for {expected} upstream connection(s) "
                    f"on {self.ip}:{self.port}")

        got = 0
        while got < expected:
            sock, addr = self.sock_server.accept()
            msg = self.recv_msg(sock, "HELLO_CLIENT_SLOT")
            payload = msg[1]
            slot_id = int(payload["slot_id"])
            cycle_id = payload.get("cycle_id", -1)
            if cycle_id is not None and int(cycle_id) >= 0:
                self.up_sock_by_cycle[(int(cycle_id), slot_id)] = sock
            else:
                self.up_sock_by_slot[slot_id] = sock

            if payload.get("run_id") is None:
                # Tier 1 client: it is waiting for the run id
                self.send_msg(sock, ["RUN_ID", self.run_id])
            else:
                self.run_id = payload["run_id"]
                self.enable_net_metrics(self.run_id)

            got += 1
            logger.info(f"[stage{self.stage_id}] upstream connected: slot={slot_id} "
                        f"cycle={cycle_id} ({got}/{expected})")

        logger.info(f"[stage{self.stage_id}] all upstream slots connected.")

    def _up(self, slot_id: int, cycle: int):
        """Upstream socket of a slot; with --clients_by_cycle each cycle has its own."""
        if self.up_sock_by_cycle:
            return self.up_sock_by_cycle[(int(cycle), int(slot_id))]
        return self.up_sock_by_slot[slot_id]

    def connect_downstream(self):
        """Open group_size persistent sockets to the next hop."""
        target = "orchestrator" if self.is_last else f"stage{self.stage_id + 1}"
        logger.info(f"[stage{self.stage_id}] opening {self.group_size} sockets to "
                    f"{target} at {self.next_ip}:{self.next_port}")

        for slot in range(self.group_size):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.next_ip, self.next_port))
            if self.is_last:
                # the orchestrator expects its own handshake, unchanged
                self.send_msg(s, ["HELLO_DATA_SLOT", {"slot_id": slot}])
            else:
                self.send_msg(s, ["HELLO_CLIENT_SLOT",
                                  {"slot_id": slot, "run_id": self.run_id}])
            self.down_sock_by_slot[slot] = s

        logger.info(f"[stage{self.stage_id}] downstream sockets ready.")

    def _init_models_once(self):
        logger.info(f"[stage{self.stage_id}] loading shard for {self.group_size} slots (once).")
        snapshot_start = helpers.get_local_resource_snapshot(self.device)

        for slot in range(self.group_size):
            model = ModelFactory.get_model_shard(
                model_name=self.model_name,
                model_type=self.model_type,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                is_first=False,
                is_last=False,
                device=self.device,
                hf_token=HF_TOKEN
            )
            model.train()
            self.models_by_slot[slot] = model
            self.opts_by_slot[slot] = optim.AdamW(model.parameters(), lr=self.lr)

        snapshot_end = helpers.get_local_resource_snapshot(self.device)
        helpers.log_snapshots(
            node_id=self.node_id, run_id=self.run_id,
            snapshot_start=snapshot_start, snapshot_end=snapshot_end,
            r=-1, i=-1, description="Model_Shard_Load"
        )

    def _reset_optimizer_for_cycle(self, slot: int):
        self.opts_by_slot[slot] = optim.AdamW(self.models_by_slot[slot].parameters(), lr=self.lr)

    def _log_peak_vram(self, cycle: int):
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        peak_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        peak_res = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
        budget = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        for name, value in (("peak_allocated_gib", peak_alloc),
                            ("peak_reserved_gib", peak_res),
                            ("vram_budget_gib", budget)):
            helpers.log_debug_metric(
                node_id=self.node_id, run_id=self.run_id, client_id="ALL",
                round_idx=cycle, context="PEAK_VRAM", entity=str(self.device),
                metric_name=name, value=value
            )
        logger.info(f"[stage{self.stage_id}] cycle {cycle} peak VRAM: "
                    f"allocated={peak_alloc:.2f} reserved={peak_res:.2f} / {budget:.2f} GiB")

    # ------------------------------------------------------------------
    # Per-slot pipeline for one cycle
    # ------------------------------------------------------------------
    def _run_slot_cycle(self, slot_id: int, cycle: int, iterations: int):
        client_id = self.slot_to_client.get(slot_id, f"slot_{slot_id}")
        up = self._up(slot_id, cycle)
        down = self.down_sock_by_slot[slot_id]

        model = self.models_by_slot[slot_id]
        opt = self.opts_by_slot[slot_id]
        model.train()

        if self.is_first:
            # release the client for this cycle
            self.send_msg(up, ["START_CYCLE", {"cycle": cycle, "client_id": client_id}])

        desc = f"[S{self.stage_id}|slot{slot_id}|{client_id}|C{cycle}]"
        for i in tqdm.tqdm(range(iterations), desc=desc):
            msg = self.recv_msg(up, "FORWARD_DATA")
            payload = msg[1]
            req_id = payload["req_id"]

            x = payload["data"].to(self.device)
            y = payload["labels"]
            am = payload.get("attention_mask")
            pi = payload.get("position_ids")
            am_dev = am.to(self.device) if am is not None else None
            pi_dev = pi.to(self.device) if pi is not None else None

            if not x.requires_grad:
                x.requires_grad_(True)

            opt.zero_grad(set_to_none=True)

            t0 = time.perf_counter()
            out = model(x, attention_mask=am_dev, position_ids=pi_dev)
            helpers.sync_device(self.device)

            t1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
                metric_name="T_comp_forward", duration=(t1 - t0),
                round_idx=cycle, batch_idx=i, req_id=req_id, delay=None
            )

            # labels and masks travel with the activations: the tail needs them
            payload_down = {
                "req_id": req_id,
                "data": out.detach().cpu(),
                "labels": y,
                "attention_mask": am,
                "position_ids": pi,
            }
            c0 = time.perf_counter()
            self.send_msg(down, ["FORWARD_DATA", payload_down])
            c1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
                metric_name="T_comm_forward", duration=(c1 - c0),
                round_idx=cycle, batch_idx=i, req_id=req_id, delay=None
            )

            msg2 = self.recv_msg(down, "BACKWARD_DATA")
            g = msg2[1]["grad"].to(self.device).to(dtype=out.dtype)

            b0 = time.perf_counter()
            out.backward(g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_max_norm)
            opt.step()
            helpers.sync_device(self.device)

            b1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
                metric_name="T_comp_backward", duration=(b1 - b0),
                round_idx=cycle, batch_idx=i, req_id=req_id, delay=None
            )

            grad_up = x.grad.detach().cpu()
            x.grad = None

            cb0 = time.perf_counter()
            self.send_msg(up, ["BACKWARD_DATA", {"req_id": req_id, "grad": grad_up}])
            cb1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
                metric_name="T_comm_backward", duration=(cb1 - cb0),
                round_idx=cycle, batch_idx=i, req_id=req_id, delay=None
            )

            del out, payload_down, msg2, g, grad_up
            del x, y, am, pi, am_dev, pi_dev

        if self.is_first:
            self.recv_msg(up, "CYCLE_DONE")
        logger.info(f"[stage{self.stage_id}] slot={slot_id} {client_id} cycle={cycle} done.")

    # ------------------------------------------------------------------
    # Cycle configuration
    # ------------------------------------------------------------------
    def _collect_cycle_config(self, cycle: int):
        """
        Stage 0 learns the cycle layout from the clients and announces it to the
        orchestrator; the remaining stages learn it from their previous hop.
        """
        self.slot_to_client = {}
        self.slot_to_iters = {}

        if self.is_first:
            for slot in range(self.group_size):
                msg = self.recv_msg(self._up(slot, cycle), "CLIENT_CONFIG")
                payload = msg[1]
                assert int(payload["cycle"]) == cycle, \
                    f"Cycle mismatch slot={slot} expected={cycle} got={payload['cycle']}"
                self.slot_to_client[slot] = payload.get("client_id", f"slot_{slot}")
                self.slot_to_iters[slot] = int(payload.get("iterations", 0))
                self.samples_by_slot[slot] = (self.samples_by_slot.get(slot, 0)
                                              + self.slot_to_iters[slot] * int(self.args.batch_size))

            self.send_msg(self.control_sock, ["START_CYCLE", {
                "cycle": cycle,
                "slot_to_client": self.slot_to_client,
                "slot_to_iters": self.slot_to_iters,
            }])
            self.recv_msg(self.control_sock, "CYCLE_READY")
        else:
            for slot in range(self.group_size):
                msg = self.recv_msg(self._up(slot, cycle), "CYCLE_CONFIG")
                payload = msg[1]
                assert int(payload["cycle"]) == cycle, \
                    f"Cycle mismatch slot={slot} expected={cycle} got={payload['cycle']}"
                self.slot_to_client[slot] = payload.get("client_id", f"slot_{slot}")
                self.slot_to_iters[slot] = int(payload.get("iterations", 0))
                self.samples_by_slot[slot] = (self.samples_by_slot.get(slot, 0)
                                              + self.slot_to_iters[slot] * int(self.args.batch_size))

        # the orchestrator already has the layout via the control plane
        if not self.is_last:
            for slot in range(self.group_size):
                self.send_msg(self.down_sock_by_slot[slot], ["CYCLE_CONFIG", {
                    "cycle": cycle,
                    "client_id": self.slot_to_client.get(slot, f"slot_{slot}"),
                    "iterations": self.slot_to_iters.get(slot, 0),
                }])

        logger.info(f"[stage{self.stage_id}] cycle {cycle}: "
                    f"clients={self.slot_to_client} iters={self.slot_to_iters}")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def start(self):
        logger.info(f"[stage{self.stage_id}] starting: total_clients={self.total_clients}, "
                    f"group_size={self.group_size}, num_cycles={self.num_cycles}")

        self._init_models_once()
        total_t0 = time.perf_counter()

        for cycle in range(self.num_cycles):
            self._collect_cycle_config(cycle)

            for slot in range(self.group_size):
                self._reset_optimizer_for_cycle(slot)

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)

            threads = []
            for slot in range(self.group_size):
                iters = int(self.slot_to_iters.get(slot, 0))
                if iters <= 0:
                    if self.is_first:
                        # keep the idle client in step with the barrier
                        sock = self._up(slot, cycle)
                        self.send_msg(sock, ["START_CYCLE", {
                            "cycle": cycle,
                            "client_id": self.slot_to_client.get(slot, f"slot_{slot}")}])
                        self.recv_msg(sock, "CYCLE_DONE")
                    continue

                t = threading.Thread(target=self._run_slot_cycle,
                                     args=(slot, cycle, iters),
                                     name=f"S{self.stage_id}-slot{slot}")
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            self._log_peak_vram(cycle)

            if self.is_first:
                self.recv_msg(self.control_sock, "CYCLE_DONE")

            logger.info(f"[stage{self.stage_id}] cycle {cycle} done.")

        total_t1 = time.perf_counter()
        helpers.log_performance_metric(
            node_id=self.node_id, client_id="ALL", seed=self.seed, run_id=self.run_id,
            metric_name="T_total_train_time", duration=(total_t1 - total_t0),
            round_idx=-1, batch_idx=-1, req_id=None, delay=None
        )

        logger.info(f"[stage{self.stage_id}] finished. Total time={(total_t1 - total_t0):.2f}s")

        self.sync_weights()

    # ------------------------------------------------------------------
    # Weight synchronization (absent from the published baseline)
    # ------------------------------------------------------------------
    def _client_socks(self, slot_id: int):
        """Client sockets of a slot: one per cycle with --clients_by_cycle, one otherwise."""
        if self.up_sock_by_cycle:
            return [self.up_sock_by_cycle[(c, slot_id)] for c in range(self.num_cycles)
                    if (c, slot_id) in self.up_sock_by_cycle]
        return [self.up_sock_by_slot[slot_id]]

    def sync_weights(self):
        """
        Collects upstream shards, adds this stage's own and passes everything to the
        next hop; then takes the averaged shards coming back, loads its own and
        relays the rest upstream. Slots are handled in the same order everywhere,
        so the chain never waits on itself.
        """
        logger.info(f"[stage{self.stage_id}] === Weight synchronization ===")

        if self.is_first and not self.recv_msg(self.control_sock, wsync.REQUEST):
            logger.error(f"[stage{self.stage_id}] no weight request; synchronization skipped.")
            return

        t_local = 0.0
        for slot in range(self.group_size):
            down = self.down_sock_by_slot[slot]

            if self.is_first:
                collected = []
                for sock in self._client_socks(slot):
                    self.send_msg(sock, [wsync.REQUEST, {"slot_id": slot}])
                    msg = self.recv_msg(sock, wsync.RESPONSE)
                    if msg:
                        collected.append(msg[1])
            else:
                msg = self.recv_msg(self.up_sock_by_slot[slot], wsync.RESPONSE_TAGGED)
                collected = list(msg[1].get("packs", [])) if msg else []

            t0 = time.perf_counter()
            collected.append(wsync.pack(self.models_by_slot[slot], f"slot_{slot}", "t2",
                                        self.start_layer, self.end_layer,
                                        samples=self.samples_by_slot.get(slot, 0)))
            t_local += time.perf_counter() - t0

            self.send_msg(down, [wsync.RESPONSE_TAGGED, {"slot_id": slot, "packs": collected}])

        for slot in range(self.group_size):
            down = self.down_sock_by_slot[slot]

            msg = self.recv_msg(down, wsync.UPDATE_TAGGED)
            averaged = list(msg[1].get("packs", [])) if msg else []

            t0 = time.perf_counter()
            wsync.apply_to(self.models_by_slot[slot],
                           wsync.match(averaged, "t2", self.start_layer, self.end_layer))
            t_local += time.perf_counter() - t0

            if self.is_first:
                for sock in self._client_socks(slot):
                    self.send_msg(sock, [wsync.UPDATE, wsync.first_of_tier(averaged, "t1")])
            else:
                self.send_msg(self.up_sock_by_slot[slot],
                              [wsync.UPDATE_TAGGED,
                               {"slot_id": slot, "packs": wsync.upstream_of(averaged, self.start_layer)}])

        helpers.log_performance_metric(
            node_id=self.node_id, client_id="ALL", seed=self.seed, run_id=self.run_id,
            metric_name="T_sync_local", duration=t_local, round_idx=-1,
            batch_idx=-1, req_id=None, delay=None)
        logger.info(f"[stage{self.stage_id}] synchronization done (local work {t_local:.2f}s)")

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    stage = HSFLChainStage(args)

    if stage.is_first:
        stage.connect_control_to_orch()
    stage.accept_upstream()
    stage.connect_downstream()
    stage.start()
