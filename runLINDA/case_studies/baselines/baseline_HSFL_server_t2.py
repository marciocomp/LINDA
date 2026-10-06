import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time
import argparse
import sys
import datetime
import gc
import socket
import tqdm

sys.path.append('../../../../')
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline HSFL T2 - Persistent connections, cycles")
    parser.add_argument("--ip", type=str, default="130.92.70.22")
    parser.add_argument("--port", type=int, default=9100)  # T1->T2
    parser.add_argument("--ip_orch", type=str, default="130.92.70.8")
    parser.add_argument("--ports_orch", type=int, default=[10101, 10102])  # [control, data]
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--model_type", type=str, default="llm")
    parser.add_argument("--split_point_start", type=int, default=1)
    parser.add_argument("--split_point_end", type=int, default=11)
    parser.add_argument("--total_layers", type=int, default=18)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--total_clients", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=2)

    return parser.parse_args()


class HSFLTier2Server(CommunicationModule):
    def __init__(self, args):
        super().__init__("server_0", args.ip)
        self.args = args

        self.ip = args.ip
        self.port = args.port

        self.ip_orch = args.ip_orch
        self.port_orch_control = args.ports_orch[0]
        self.port_orch_data = args.ports_orch[1]

        self.seed = args.seed
        self.lr = args.lr

        self.model_name = args.model_name
        self.model_type = args.model_type
        self.start_layer = args.split_point_start
        self.end_layer = args.split_point_end

        self.total_clients = int(args.total_clients)
        self.group_size = int(args.group_size)
        self.num_cycles = (self.total_clients + self.group_size - 1) // self.group_size

        self.device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

        self.run_id = None
        self.control_sock = None  # T2 -> T3 control

        self.client_sock_by_slot = {}     # slot_id -> socket (T1->T2)
        self.orch_sock_by_slot = {}       # slot_id -> socket (T2->T3 data)

        self.slot_to_client = {}
        self.slot_to_iters = {}

        self.models_by_slot = {}
        self.opts_by_slot = {}

        self.clip_max_norm = float(args.clip_max_norm)

        self.initial_snapshot = helpers.get_local_resource_snapshot(self.device)

    # ----------------------------
    # Connect to T3 control
    # ----------------------------
    def do_connect_to_orch(self):
        self.sock_client.connect((self.ip_orch, self.port_orch_control))
        self.control_sock = self.sock_client
        self.send_msg(self.control_sock, ["HELLO_TIER2"])
        msg = self.recv_msg(self.control_sock, "RUN_ID")
        self.run_id = msg[1]
        logger.info(f"[T2] Connected to Orch. run_id={self.run_id}")

    # ----------------------------
    # Accept group_size client sockets once
    # ----------------------------
    def accept_slots_from_clients(self):
        self.sock_server.bind((self.ip, self.port))
        self.sock_server.listen(20)
        logger.info(f"[T2] Listening for {self.group_size} client slots on {self.ip}:{self.port}")

        got = 0
        while got < self.group_size:
            sock, addr = self.sock_server.accept()
            msg = self.recv_msg(sock, "HELLO_CLIENT_SLOT")
            slot_id = int(msg[1]["slot_id"])
            self.client_sock_by_slot[slot_id] = sock
            got += 1
            logger.info(f"[T2] Client slot connected: slot={slot_id} ({got}/{self.group_size})")

        # envia RUN_ID uma vez por slot (fixo)
        for slot, sock in self.client_sock_by_slot.items():
            self.send_msg(sock, ["RUN_ID", self.run_id])

        logger.info("[T2] All client slots connected.")

    # ----------------------------
    # Open group_size data sockets to T3 once
    # ----------------------------
    def connect_slots_to_orch_data(self):
        logger.info(f"[T2] Opening {self.group_size} persistent data sockets to T3...")
        for slot in range(self.group_size):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.ip_orch, self.port_orch_data))
            self.send_msg(s, ["HELLO_DATA_SLOT", {"slot_id": slot}])
            self.orch_sock_by_slot[slot] = s
        logger.info("[T2] All T2->T3 data slots ready.")

    # ----------------------------
    # Init models per slot once
    # ----------------------------
    def _init_models_once(self):
        logger.info(f"[T2] Loading intermediate shard models for {self.group_size} slots (once).")
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
            node_id=self.node_id,
            run_id=self.run_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            r=-1,
            i=-1,
            description="Model_Shard_Load"
        )

    def _reset_optimizer_for_cycle(self, slot: int):
        model = self.models_by_slot[slot]
        self.opts_by_slot[slot] = optim.AdamW(model.parameters(), lr=self.lr)

    # ----------------------------
    # Per-slot pipeline for one cycle
    # ----------------------------
    def _run_slot_cycle(self, slot_id: int, cycle: int, iterations: int):
        client_id = self.slot_to_client.get(slot_id, f"slot_{slot_id}")
        cs = self.client_sock_by_slot[slot_id]
        osock = self.orch_sock_by_slot[slot_id]

        model = self.models_by_slot[slot_id]
        opt = self.opts_by_slot[slot_id]
        model.train()

        self.send_msg(cs, ["START_CYCLE", {"cycle": cycle, "client_id": client_id}])

        for i in tqdm.tqdm(range(iterations), desc=f"[T2|slot{slot_id}|{client_id}|C{cycle}]"):
            msg = self.recv_msg(cs, "FORWARD_DATA")
            payload = msg[1]
            req_id = payload["req_id"]

            x = payload["data"].to(self.device)
            y = payload["labels"].to(self.device)
            am = payload.get("attention_mask")
            pi = payload.get("position_ids")
            am = am.to(self.device) if am is not None else None
            pi = pi.to(self.device) if pi is not None else None

            if not x.requires_grad:
                x.requires_grad_(True)

            opt.zero_grad(set_to_none=True)

            # forward local shard
            t0 = time.perf_counter()
            out = model(x, attention_mask=am, position_ids=pi)
            helpers.sync_device(self.device)

            t1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
                metric_name="T_comp_forward", duration=(t1 - t0),
                round_idx=cycle, batch_idx=i, req_id=req_id, delay=None
            )

            # send to T3
            payload_t3 = {
                "req_id": req_id,
                "data": out.detach().cpu(),
                "labels": y.cpu(),
                "attention_mask": am.cpu() if am is not None else None,
                "position_ids": pi.cpu() if pi is not None else None,
            }
            c0 = time.perf_counter()
            self.send_msg(osock, ["FORWARD_DATA", payload_t3])
            c1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
                metric_name="T_comm_forward", duration=(c1 - c0),
                round_idx=cycle, batch_idx=i, req_id=req_id, delay=None
            )

            # recv grad from T3
            msg2 = self.recv_msg(osock, "BACKWARD_DATA")
            g = msg2[1]["grad"].to(self.device)

            # backward local shard
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

            # send grad to client
            grad_to_client = x.grad.detach().cpu()
            x.grad = None
            resp = {"req_id": req_id, "grad": grad_to_client}

            cb0 = time.perf_counter()
            self.send_msg(cs, ["BACKWARD_DATA", resp])
            cb1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=self.node_id, client_id=client_id, seed=self.seed, run_id=self.run_id,
                metric_name="T_comm_backward", duration=(cb1 - cb0),
                round_idx=cycle, batch_idx=i, req_id=req_id, delay=None
            )

            del out, payload_t3, msg2, g, resp, grad_to_client
            del x, y, am, pi

        # fim do ciclo do slot
        self.recv_msg(cs, "CYCLE_DONE")  # cliente manda CYCLE_DONE
        logger.info(f"[T2] slot={slot_id} {client_id} cycle={cycle} done.")

    # ----------------------------
    # Main: cycles
    # ----------------------------
    def start(self):
        logger.info(f"[T2] Starting cycles: total_clients={self.total_clients}, group_size={self.group_size}, num_cycles={self.num_cycles}")

        self._init_models_once()

        total_t0 = time.perf_counter()

        for cycle in range(self.num_cycles):
            self.slot_to_client = {}
            self.slot_to_iters = {}

            for slot in range(self.group_size):
                sock = self.client_sock_by_slot[slot]
                msg = self.recv_msg(sock, "CLIENT_CONFIG")
                payload = msg[1]
                assert int(payload["cycle"]) == cycle, f"Cycle mismatch slot={slot} expected={cycle} got={payload['cycle']}"
                self.slot_to_client[slot] = payload.get("client_id", f"slot_{slot}")
                self.slot_to_iters[slot] = int(payload.get("iterations", 0))

            logger.info(f"[T2] Cycle {cycle}: slot_to_client={self.slot_to_client} slot_to_iters={self.slot_to_iters}")

            self.send_msg(self.control_sock, ["START_CYCLE", {"cycle": cycle, "slot_to_client": self.slot_to_client, "slot_to_iters": self.slot_to_iters}])
            _ = self.recv_msg(self.control_sock, "CYCLE_READY")

            for slot in range(self.group_size):
                self._reset_optimizer_for_cycle(slot)

            threads = []
            for slot in range(self.group_size):
                iters = int(self.slot_to_iters.get(slot, 0))
                if iters <= 0:
                    # ainda assim manda START_CYCLE para o cliente não travar
                    self.send_msg(self.client_sock_by_slot[slot], ["START_CYCLE", {"cycle": cycle, "client_id": self.slot_to_client.get(slot, f"slot_{slot}")}])
                    # e espera CYCLE_DONE dele
                    self.recv_msg(self.client_sock_by_slot[slot], "CYCLE_DONE")
                    continue

                t = threading.Thread(target=self._run_slot_cycle, args=(slot, cycle, iters), name=f"T2-slot{slot}")
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            _ = self.recv_msg(self.control_sock, "CYCLE_DONE")
            logger.info(f"[T2] Cycle {cycle} done.")

        total_t1 = time.perf_counter()
        helpers.log_performance_metric(
            node_id=self.node_id,
            client_id="ALL",
            seed=self.seed,
            run_id=self.run_id,
            metric_name="T_total_train_time",
            duration=(total_t1 - total_t0),
            round_idx=-1,
            batch_idx=-1,
            req_id=None,
            delay=None
        )

        logger.info(f"[T2] Finished. Total time={(total_t1-total_t0):.2f}s")

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    t2 = HSFLTier2Server(args)
    t2.do_connect_to_orch()
    t2.accept_slots_from_clients()
    t2.connect_slots_to_orch_data()
    t2.start()