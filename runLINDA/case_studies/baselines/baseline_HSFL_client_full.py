import torch
import torch.optim as optim
import time
import argparse
import os
import sys
from pathlib import Path
import gc
import uuid
import tqdm
import math

sys.path.append('../../../../..')
sys.path.append(str(Path(__file__).resolve().parents[5]))   # repo root, whatever the cwd


from torch.utils.data import DataLoader
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils.datasets.alpaca import AlpacaDataset, collate_fn
from utils.datasets.data_partitioner import DataPartitioner
from utils import helpers
import weights_sync_full as wsync


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline HSFL Client (T1) - Persistent connection, cycles")
    parser.add_argument("--slot_id", type=int, required=True, )
    parser.add_argument("--cycle_id", type=int, default=-1,
                        help="cycle this process serves; with one process per slot AND cycle the "
                             "clients of a cycle sit on the same host and contend for it, as the "
                             "Tier 1 agents of LINDA do. -1 keeps one process serving every cycle")
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--total_clients", type=int, default=4)

    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--ip_server", type=str, default="130.92.70.8")
    parser.add_argument("--port_server", type=int, default=20100)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--model_type", type=str, default="llm")
    parser.add_argument("--split_point", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)

    parser.add_argument("--data_dir", type=str, default="~/LINDA_data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--new_size", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)

    parser.add_argument("--max_steps", type=int, default=0, help="0 => use actual iterations; >0 limits batches per cycle")
    return parser.parse_args()


class HSFLClientSlot(CommunicationModule):
    def __init__(self, args):
        cycle_id = int(getattr(args, "cycle_id", -1))
        node_id = (f"client_slot_{args.slot_id}" if cycle_id < 0
                   else f"client_slot_{args.slot_id}_c{cycle_id}")
        super().__init__(node_id, args.ip)
        self.args = args

        self.slot_id = int(args.slot_id)
        self.group_size = int(args.group_size)
        self.samples = 0          # everything this slot trained, for the FedAvg weight
        self.cycle_id = cycle_id  # -1: every cycle; otherwise only this one
        self.total_clients = int(args.total_clients)
        self.num_cycles = (self.total_clients + self.group_size - 1) // self.group_size

        self.server_ip = args.ip_server
        self.server_port = args.port_server

        self.seed = args.seed
        self.alpha = args.alpha
        self.lr = args.lr
        self.model_name = args.model_name
        self.model_type = args.model_type
        self.end_layer = int(args.split_point)

        self.new_size = args.new_size
        self.data_dir = args.data_dir
        self.batch_size = int(args.batch_size)
        self.max_steps = int(args.max_steps)
        self.clip_max_norm = float(args.clip_max_norm)

        if args.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.run_id = None

        # persistent slot model
        self.model = ModelFactory.get_model_shard(
            model_name=self.model_name,
            model_type=self.model_type,
            start_layer=0,
            end_layer=self.end_layer,
            is_first=True,
            is_last=False,
            device=self.device,
            hf_token=HF_TOKEN
        )
        self.model.train()

    def connect_once(self):
        logger.info(f"[slot{self.slot_id}] Connecting once to T2 at {self.server_ip}:{self.server_port}...")
        self.sock_client.connect((self.server_ip, self.server_port))

        self.send_msg(self.sock_client, ["HELLO_CLIENT_SLOT",
                                         {"slot_id": self.slot_id, "cycle_id": self.cycle_id}])

        msg = self.recv_msg(self.sock_client, "RUN_ID")
        self.run_id = msg[1]
        self.enable_net_metrics(self.run_id)
        logger.info(f"[slot{self.slot_id}] run_id={self.run_id}")

    def _prepare_loader_for_logical_client(self, client_training_id: int):
        dataset_name = "tatsu-lab/alpaca"
        collate = collate_fn

        eval_indices, train_pool_indices, universe_indices = DataPartitioner.get_global_eval_and_train_pool(
            dataset_name=dataset_name,
            new_size=self.new_size,
            eval_fraction=0.10,
            seed=self.seed,
            cache_dir=self.data_dir
        )

        train_indices = DataPartitioner.get_partition_indices(
            dataset_name=dataset_name,
            total_clients=self.total_clients,
            partition_id=client_training_id,
            method="content_dirichlet_balanced",
            alpha=self.alpha,
            seed=self.seed,
            cache_dir=self.data_dir,
            pool_indices=train_pool_indices
        )

        train_dataset = AlpacaDataset(
            tokenizer_name=self.model_name,
            split="train",
            indices=train_indices,
            cache_dir=self.data_dir
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True
        )
        iterations = len(train_loader)
        if self.max_steps > 0:
            iterations = min(iterations, self.max_steps)

        return train_loader, iterations

    def _cycle_client_id(self, cycle: int):
        idx = self.slot_id + cycle * self.group_size
        if idx >= self.total_clients:
            return None, None
        return f"client_{idx}", idx

    def run(self):
        cycles = range(self.num_cycles) if self.cycle_id < 0 else [self.cycle_id]
        for cycle in cycles:
            logical_client_id, logical_idx = self._cycle_client_id(cycle)

            if logical_client_id is None:
                # slot has no work this cycle
                self.send_msg(self.sock_client, ["CLIENT_CONFIG", {"cycle": cycle, "client_id": f"client_skip_{self.slot_id}", "iterations": 0}])
                _ = self.recv_msg(self.sock_client, "START_CYCLE")
                self.send_msg(self.sock_client, ["CYCLE_DONE", {"cycle": cycle, "client_id": f"client_skip_{self.slot_id}"}])
                continue

            train_loader, iterations = self._prepare_loader_for_logical_client(logical_idx)
            self.samples += iterations * int(self.args.batch_size)

            self.send_msg(self.sock_client, ["CLIENT_CONFIG", {"cycle": cycle, "client_id": logical_client_id, "iterations": iterations}])

            # wait for T2 start barrier
            _ = self.recv_msg(self.sock_client, "START_CYCLE")

            # reinit optimizer
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            self.model.train()

            snapshot_start = helpers.get_local_resource_snapshot(self.device)

            t_cycle0 = time.perf_counter()

            for i, batch in enumerate(tqdm.tqdm(train_loader, total=iterations, desc=f"[{logical_client_id}|slot{self.slot_id}|C{cycle}]")):
                if i >= iterations:
                    break

                req_id = str(uuid.uuid4())
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                optimizer.zero_grad(set_to_none=True)

                # forward local shard
                t0 = time.perf_counter()
                out = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
                helpers.sync_device(self.device)

                t1 = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=logical_client_id,
                    client_id=logical_client_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_forward",
                    duration=(t1 - t0),
                    round_idx=cycle,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                payload = {
                    "req_id": req_id,
                    "data": out.detach().cpu(),
                    "labels": labels.cpu(),
                    "attention_mask": attention_mask.cpu(),
                    "position_ids": position_ids.cpu(),
                }

                c0 = time.perf_counter()
                self.send_msg(self.sock_client, ["FORWARD_DATA", payload])
                c1 = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=logical_client_id,
                    client_id=logical_client_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comm_forward",
                    duration=(c1 - c0),
                    round_idx=cycle,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                msg = self.recv_msg(self.sock_client, "BACKWARD_DATA")
                grad = msg[1]["grad"].to(self.device).to(dtype=out.dtype)

                b0 = time.perf_counter()
                out.backward(grad)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
                optimizer.step()
                helpers.sync_device(self.device)

                b1 = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=logical_client_id,
                    client_id=logical_client_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_backward",
                    duration=(b1 - b0),
                    round_idx=cycle,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                snapshot_end = helpers.get_local_resource_snapshot(self.device)
                helpers.log_snapshots(
                    node_id=logical_client_id,
                    run_id=self.run_id,
                    snapshot_start=snapshot_start,
                    snapshot_end=snapshot_end,
                    r=cycle,
                    i=i,
                    description="During_train"
                )

                del out, grad, msg, payload
                del input_ids, attention_mask, labels, position_ids

            t_cycle1 = time.perf_counter()
            helpers.log_performance_metric(
                node_id=logical_client_id,
                client_id=logical_client_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_cycle_total",
                duration=(t_cycle1 - t_cycle0),
                round_idx=cycle,
                batch_idx=iterations,
                req_id=None,
                delay=None
            )

            # barreira do ciclo
            self.send_msg(self.sock_client, ["CYCLE_DONE", {"cycle": cycle, "client_id": logical_client_id}])

            # limpeza leve
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            del optimizer
            gc.collect()

        logger.info(f"[slot{self.slot_id}] All cycles completed (cycles={list(cycles)}).")
        self.sync_weights()

    # ------------------------------------------------------------------
    # Weight synchronization (absent from the published baseline)
    # ------------------------------------------------------------------
    def sync_weights(self):
        """Sends this slot's shard up the chain and loads back the averaged one."""
        logger.info(f"[slot{self.slot_id}] === Weight synchronization ===")
        if not self.recv_msg(self.sock_client, wsync.REQUEST):
            logger.error(f"[slot{self.slot_id}] no weight request; synchronization skipped.")
            return

        t0 = time.perf_counter()
        pack = wsync.pack(self.model, f"slot_{self.slot_id}", "t1", 0, self.end_layer,
                          samples=self.samples)
        t_pack = time.perf_counter() - t0
        self.send_msg(self.sock_client, [wsync.RESPONSE, pack])

        msg = self.recv_msg(self.sock_client, wsync.UPDATE)
        t1 = time.perf_counter()
        applied = wsync.apply_to(self.model, msg[1]) if msg else False
        t_apply = time.perf_counter() - t1

        helpers.log_performance_metric(
            node_id=self.node_id, client_id=f"slot_{self.slot_id}", seed=self.seed,
            run_id=self.run_id, metric_name="T_sync_local", duration=(t_pack + t_apply),
            round_idx=-1, batch_idx=-1, req_id=None, delay=None)
        logger.info(f"[slot{self.slot_id}] synchronization done (applied={applied}, "
                    f"pack {t_pack:.2f}s, apply {t_apply:.2f}s)")


if __name__ == "__main__":
    args = parse_args()
    cli = HSFLClientSlot(args)
    cli.connect_once()
    cli.run()