import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time
import argparse
import sys
import datetime
import gc
import tqdm

sys.path.append('../../../../')
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers

from offload_engine import CPUOffloadEngine, estimate_host_gib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline HSFL Orch (T3) WITH host-memory offloading - persistent connections, cycles")
    parser.add_argument("--ip", type=str, default="130.92.70.8")
    parser.add_argument("--ports", type=int, default=[10101, 10102])  # [control, data]
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--model_type", type=str, default="llm")
    parser.add_argument("--split_point", type=int, default=11)
    parser.add_argument("--total_layers", type=int, default=18)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--total_clients", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)

    # ---- host-memory offloading ----
    parser.add_argument("--offload_optimizer", type=int, default=1,
                        help="1: master weights, Adam moments and grads in host RAM")
    parser.add_argument("--offload_params", type=int, default=1,
                        help="1: stream decoder-layer and LM-head weights in/out of the GPU")
    parser.add_argument("--master_dtype", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="precision of the CPU master weights and Adam moments")
    parser.add_argument("--pin_memory", type=int, default=1,
                        help="1: pin the CPU masters (faster PCIe, non-swappable host RAM)")
    return parser.parse_args()


_MASTER_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class HSFLTier3OrchOffload(CommunicationModule):
    def __init__(self, args):
        super().__init__("server_orch", args.ip)

        self.args = args
        self.run_id = f"b_exp2_HSFL_offload_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

        self.ip = args.ip
        self.port_control = args.ports[0]
        self.port_data = args.ports[1]

        self.seed = args.seed
        self.lr = args.lr

        self.model_name = args.model_name
        self.model_type = args.model_type
        self.start_layer = args.split_point
        self.end_layer = args.total_layers

        self.total_clients = int(args.total_clients)
        self.group_size = int(args.group_size)
        self.num_cycles = (self.total_clients + self.group_size - 1) // self.group_size

        self.device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
        if torch.cuda.is_available():
            gpu_id = int(args.device.split(":")[1])
            torch.cuda.set_device(gpu_id)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # offload config
        self.offload_optimizer = bool(int(args.offload_optimizer))
        self.offload_params = bool(int(args.offload_params))
        self.master_dtype = _MASTER_DTYPES[args.master_dtype]
        self.pin_memory = bool(int(args.pin_memory))

        self.control_sock = None          # T2 control socket
        self.data_socks_by_slot = {}      # slot_id -> socket (T2->T3)
        self.slot_to_client = {}          # slot_id -> client_id (por ciclo)
        self.slot_to_iters = {}           # slot_id -> iterations (por ciclo)

        self.models_by_slot = {}
        self.engines_by_slot = {}

        self.clip_max_norm = float(args.clip_max_norm)

        self.initial_snapshot = helpers.get_local_resource_snapshot(self.device)

    # ----------------------------
    # Control (T2)
    # ----------------------------
    def do_connection_control(self):
        self.sock_server.bind((self.ip, self.port_control))
        self.sock_server.listen(10)
        logger.info(f"[T3] Control listening on {self.ip}:{self.port_control}")

        sock, addr = self.sock_server.accept()
        logger.info(f"[T3] Control connected: {addr}")

        self.recv_msg(sock, "HELLO_TIER2")
        self.send_msg(sock, ["RUN_ID", self.run_id])
        self.control_sock = sock

    # ----------------------------
    # Data sockets (T2->T3):
    # ----------------------------
    def do_connection_data(self):
        self.sock_server_data.bind((self.ip, self.port_data))
        self.sock_server_data.listen(20)
        logger.info(f"[T3] Data listening on {self.ip}:{self.port_data} (expect {self.group_size} slots)")

        got = 0
        while got < self.group_size:
            sock, addr = self.sock_server_data.accept()
            msg = self.recv_msg(sock, "HELLO_DATA_SLOT")
            slot_id = int(msg[1]["slot_id"])
            self.data_socks_by_slot[slot_id] = sock
            got += 1
            logger.info(f"[T3] Data slot connected: slot={slot_id} ({got}/{self.group_size})")

        logger.info("[T3] All data slots connected.")

    # ----------------------------
    # Models per slot (persist once) + offload engines
    # ----------------------------
    def _init_models_once(self):
        logger.info(f"[T3] Loading final shard models for {self.group_size} slots (once). "
                    f"offload_opt={self.offload_optimizer} offload_par={self.offload_params}")
        snapshot_start = helpers.get_local_resource_snapshot(self.device)

        for slot in range(self.group_size):
            model = ModelFactory.get_model_shard(
                model_name=self.model_name,
                model_type=self.model_type,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                is_first=False,
                is_last=True,
                device=self.device,
                hf_token=HF_TOKEN
            )
            model.train()
            self.models_by_slot[slot] = model

            if slot == 0:
                est = estimate_host_gib(model, self.master_dtype, self.group_size)
                host = helpers.get_local_resource_snapshot(torch.device("cpu"))
                logger.info(f"[T3] Host RAM needed by offload: {est:.2f} GiB "
                            f"for {self.group_size} slots | host free: {host['ram_free_gb']:.2f} GiB")
                if self.offload_optimizer and est > host["ram_free_gb"]:
                    logger.warning("[T3] Offload footprint exceeds free host RAM. "
                                   "Consider --master_dtype bf16 or --pin_memory 0.")

            self.engines_by_slot[slot] = CPUOffloadEngine(
                model=model,
                device=self.device,
                lr=self.lr,
                offload_optimizer=self.offload_optimizer,
                offload_params=self.offload_params,
                master_dtype=self.master_dtype,
                pin_memory=self.pin_memory,
                clip_max_norm=self.clip_max_norm,
                label=f"T3-slot{slot}",
            )

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        snapshot_end = helpers.get_local_resource_snapshot(self.device)
        helpers.log_snapshots(
            node_id=self.node_id,
            run_id=self.run_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            r=-1,
            i=-1,
            description="Model_Shard_Load_Offload"
        )

    def _reset_optimizer_for_cycle(self, slot: int):

        self.engines_by_slot[slot].reset_optimizer()

    # ----------------------------
    # One slot stream handler (reads FORWARD_DATA for N iters)
    # ----------------------------
    def _calculate_loss(self, output, labels):
        """
        Metodo auxiliar para calcular Loss.
        """
        logits = output[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return self.criterion(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1)
        )

    def _log_peak_vram(self, cycle):
        if self.device.type != "cuda":
            return
        peak_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        peak_res = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
        logger.info(f"[T3] Cycle {cycle} peak VRAM: allocated={peak_alloc:.2f} GiB "
                    f"reserved={peak_res:.2f} GiB")
        helpers.log_performance_metric(
            node_id=self.node_id, client_id="ALL", seed=self.seed, run_id=self.run_id,
            metric_name="VRAM_peak_allocated_gib", duration=peak_alloc,
            round_idx=cycle, batch_idx=-1, req_id=None, delay=None
        )
        helpers.log_performance_metric(
            node_id=self.node_id, client_id="ALL", seed=self.seed, run_id=self.run_id,
            metric_name="VRAM_peak_reserved_gib", duration=peak_res,
            round_idx=cycle, batch_idx=-1, req_id=None, delay=None
        )

    def _handle_slot_cycle(self, slot_id: int, cycle: int, iterations: int):
        client_id = self.slot_to_client.get(slot_id, f"slot_{slot_id}")
        sock = self.data_socks_by_slot[slot_id]
        model = self.models_by_slot[slot_id]
        engine = self.engines_by_slot[slot_id]
        model.train()

        for i in tqdm.tqdm(range(iterations), desc=f"[T3|slot{slot_id}|{client_id}|C{cycle}]"):
            msg = self.recv_msg(sock, "FORWARD_DATA")
            payload = msg[1]
            req_id = payload.get("req_id", None)

            x = payload["data"].to(self.device)
            y = payload["labels"].to(self.device)
            am = payload.get("attention_mask")
            pi = payload.get("position_ids")
            am = am.to(self.device) if am is not None else None
            pi = pi.to(self.device) if pi is not None else None

            if not x.requires_grad:
                x.requires_grad_(True)

            engine.zero_grad()

            t0 = time.perf_counter()
            out = model(x, attention_mask=am, position_ids=pi)

            loss = self._calculate_loss(out, y)

            loss.backward()
            helpers.sync_device(self.device)
            t_bwd = time.perf_counter()

            engine.step()
            helpers.sync_device(self.device)

            t1 = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=client_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comp",
                duration=(t1 - t0),
                round_idx=cycle,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            # isolates the cost that offloading adds: the host-side optimizer step
            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=client_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_optimizer_step",
                duration=(t1 - t_bwd),
                round_idx=cycle,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            grad = x.grad.detach().cpu()
            resp = {"req_id": req_id, "grad": grad}

            c0 = time.perf_counter()
            self.send_msg(sock, ["BACKWARD_DATA", resp])
            c1 = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=client_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comm",
                duration=(c1 - c0),
                round_idx=cycle,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            del out, loss, resp, grad
            del x, y, am, pi

    # ----------------------------
    # Main cycles
    # ----------------------------
    def start(self):
        logger.info(f"[T3] Starting cycles: total_clients={self.total_clients}, "
                    f"group_size={self.group_size}, num_cycles={self.num_cycles}")

        self._init_models_once()

        total_t0 = time.perf_counter()

        for cycle in range(self.num_cycles):
            msg = self.recv_msg(self.control_sock, "START_CYCLE")
            payload = msg[1]
            assert int(payload["cycle"]) == cycle, f"Cycle mismatch: expected={cycle}, got={payload['cycle']}"

            self.slot_to_client = {int(k): v for k, v in payload["slot_to_client"].items()}
            self.slot_to_iters = {int(k): int(v) for k, v in payload["slot_to_iters"].items()}

            logger.info(f"[T3] Cycle {cycle}: slot_to_client={self.slot_to_client} slot_to_iters={self.slot_to_iters}")

            # reset optimizer por slot do ciclo
            for slot in range(self.group_size):
                self._reset_optimizer_for_cycle(slot)

            # ready
            self.send_msg(self.control_sock, ["CYCLE_READY", {"cycle": cycle}])
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
            threads = []
            for slot in range(self.group_size):
                iters = int(self.slot_to_iters.get(slot, 0))
                if iters <= 0:
                    continue
                t = threading.Thread(target=self._handle_slot_cycle,
                                     args=(slot, cycle, iters),
                                     name=f"T3-slot{slot}")
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            self._log_peak_vram(cycle)

            # done
            self.send_msg(self.control_sock, ["CYCLE_DONE", {"cycle": cycle}])
            logger.info(f"[T3] Cycle {cycle} done.")

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

        logger.info(f"[T3] Finished. Total time={(total_t1-total_t0):.2f}s")
        for slot, engine in self.engines_by_slot.items():
            logger.info(f"[T3] slot{slot} offload: {engine.report()}")

        # cleanup leve (mantém sockets)
        for engine in self.engines_by_slot.values():
            engine.close()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    orch = HSFLTier3OrchOffload(args)
    orch.do_connection_control()
    orch.do_connection_data()
    orch.start()
