# @author: Marcio Lopes
import copy
import gc
import random
import sys
import json
import threading
import time
import datetime
import socket
import transformers
import tqdm


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append('../../')
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers
from utils.standardTuples.data_payload import ForwardDataTuple, BackwardGradTuple


class ComputeNode(CommunicationModule):
    """
    Tier 2 Compute Worker. Executes one segment of the local chain, between its
    previous hop and the next.
    """

    def __init__(self,
                 node_id,
                 ip_address,
                 local_port,
                 manager_ip,
                 manager_port,
                 manager_port_data,
                 orchestrator_ports,
                 device,
                 proxies=None):
        super().__init__(node_id, ip_address)

        self.proxies = proxies or {}
        self.weight_decay = None
        self.fedprox_cpu = None
        self.alpha = None
        self.mu = 0.0
        self.client_global_params = {}
        self.manager_sock_data = {}
        self.clip_max_norm = None
        self.run_id = None
        self.start_layer = None
        self.end_layer = None
        self.threads_evaluate_clients = None
        self.learning_rate = None
        self.base_models_per_client = {}
        self.optimizers = {}
        self.threads_training_clients = None
        self.next_hop = None
        self.next_ip = None
        self.next_port = None
        self.clients = None
        self.iterations_train = None
        self.iterations_eval = None
        self.context_timestamps = []
        self.first_task = None
        self.seed = None
        self.local_port = local_port
        self.manager_ip = manager_ip
        self.manager_port = manager_port
        self.manager_port_data = manager_port_data
        self.orchestrator_ports = orchestrator_ports
        self.global_rounds = None
        self.delay = 0.0
        self.snapshot_end = {}
        self.snapshot_start = {}

        self.manager_sock = None
        self.sock_previous_hop = {}
        self.previous_hop = None
        self.sock_next_hop = {}
        self.prevhop_recv_lock = threading.Lock()
        self.prevhop_inbox = {}
        self.prevhop_eval_lock = threading.Lock()
        self.prevhop_eval_inbox = {}
        self.manager_send_lock = threading.Lock()

        self.pending_weight_packs = {}

        self.pending_optimizer_moments = {}

        self.tasks = {}
        if device == "cpu":
            self.device = torch.device(device)  # device #"cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if torch.cuda.is_available():
                gpu_id = int(device.split(":")[1])
                torch.cuda.set_device(gpu_id)
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")

        self.models = {}
        self.base_model = None
        self.model_name = None
        self.model_type = None
        self.hf_token = HF_TOKEN
        self.train_lock = threading.Lock()
        self.client_norms = {}
        self.should_offload = None
        self.is_tail = None
        self.local_finish = None
        self.num_classes = None

        self.criterion = None

        self.execution_context = {}
        self.context_lock = threading.Lock()
        self.my_hw = helpers.get_local_device_name(self.device)
        logger.info(f"Compute Node {node_id} initialized on {self.device} (Port: {local_port})")
        logger.info(f"Device Name: {self.my_hw}")
        self.initial_snapshot = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Initial System Snapshot : {self.initial_snapshot}")



    def _set_global_seed(self):
        """
        Sets the random seed for reproducibility across all libraries.
        """

        logger.info(f"Setting Global Seed to: {self.seed}")
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.device.type == "cuda":
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def connect_to_manager(self):
        """Connects to Cluster Manager Control Plane."""
        dial = helpers.dial_ip(self.proxies, self.ip, self.manager_ip)
        logger.info(f"Connecting to Manager at {self.manager_ip}:{self.manager_port} "
                    f"(dialing {dial}:{self.manager_port})...")
        try:
            self.sock_client.connect((dial, self.manager_port))
            self.manager_sock = self.sock_client

            # The resource vector travels with the greeting: the orchestrator needs
            # it before it can allocate, and this is the first moment this node can
            # be heard. It is measured on the device, not declared for the hostname.
            self.send_msg(self.manager_sock, ["HELLO_COMPUTE",
                                              [self.node_id, self.my_hw,
                                               helpers.measure_local_resources(
                                                   self.node_id, self.device)]])
            logger.info("Connected to Manager!")
            self.perform_time_sync_handshake(self.manager_sock, role="client")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _connect_to_next_hop(self,device_id):
        """Connects to the next node in the chain."""
        if self.next_hop == "LOCAL_FINISH":
            logger.info(f"[{self.node_id}] next_hop=LOCAL_FINISH -> no connection needed.")
            self.sock_next_hop = None
            return True

        target_port = self.next_port
        target_ip = helpers.dial_ip(self.proxies, self.ip, self.next_ip)
        target_id = self.next_hop
        while True:
            try:
                logger.info(f"Connecting to Next Hop ({target_id}): {self.next_ip}:{target_port} "
                            f"(dialing {target_ip}:{target_port})...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((target_ip, target_port))

                if target_port != self.orchestrator_ports[1]:
                    # msg = ["PREVIOUS_HOP", self.node_id]
                    msg = ["PREVIOUS_HOP", {"previous_hop": self.node_id, "device_id": device_id}]
                    self.send_msg(sock, msg)

                    if self.next_hop not in self.sock_next_hop:
                        self.sock_next_hop[self.next_hop] = {}
                    self.sock_next_hop[self.next_hop][device_id] = sock

                    logger.info("-> Connected to Next Hop.")
                return True
            except Exception as e:
                logger.error(f"Failed to connect next hop: {e}")
                time.sleep(2)
                # return False

    def _get_client_model(self, client_id):
        """
        Returns (model, optimizer) specific to the client.
        Creates a copy from the site template if needed.
        """
        with self.train_lock:

            if client_id not in self.models:
                logger.info(f"Creating Orchestrator Replica for Client: {client_id}")

                new_model, new_optimizer = self._get_local_model_shard(client_id)
                new_model.to(self.device)

                self.models[client_id] = new_model
                self.optimizers[client_id] = new_optimizer

                held = self.pending_weight_packs.pop(client_id, None)
                if held:
                    new_model.load_state_dict(held, strict=False)
                    logger.info(f"[REALLOC][{self.node_id}] applied the parameters of "
                                f"the new range to {client_id}")

                carried = self.pending_optimizer_moments.pop(client_id, None)
                if carried:
                    restored = helpers.restore_optimizer_moments(
                        new_model, new_optimizer, carried)
                    logger.info(f"[REALLOC][{self.node_id}] restored the moments of "
                                f"{restored} parameters for {client_id}")
        return self.models[client_id], self.optimizers[client_id]

    def wait_for_configuration(self):
        """
        Receives model config and chain config from Manager.
        """
        logger.info("Waiting for Task Configuration...")
        msg = self.recv_msg(self.manager_sock, "COMPUTE_CONFIG")

        if msg:
            task_list = msg[1]
            if not task_list:
                logger.warning("Received empty task list.")
                return False

            self.first_task = task_list[0]
            self.start_layer = int(self.first_task['start_layer'])
            self.end_layer = int(self.first_task['end_layer'])

            self.model_name = self.first_task['model_name']
            self.model_type = self.first_task['model_type']
            self.seed = self.first_task['seed']
            self.run_id = self.first_task['run_id']
            self.enable_net_metrics(self.run_id)
            self._set_global_seed()

            if self.model_type == "llm":
                self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
                self.clip_max_norm = 1.0
            else:
                self.criterion = nn.CrossEntropyLoss()
                self.clip_max_norm = 1.0


            # Mode flags
            self.should_offload = self.first_task.get('should_offload', True)
            self.is_tail = self.first_task.get('is_tail', False)
            self.local_finish = self.first_task.get('local_finish', False)
            self.num_classes = self.first_task.get('num_classes', 10)

            logger.info(
                f"[CFG][{self.node_id}] should_offload={self.should_offload} is_tail={self.is_tail} local_finish={self.local_finish}")

            self.next_hop = self.first_task['next_hop']
            self.next_port = self.first_task['next_port']
            self.next_ip = self.first_task['next_ip']
            self.clients = self.first_task['devices']

            if self.next_hop == "LOCAL_FINISH":
                logger.info(f"[CFG][{self.node_id}] next_hop=LOCAL_FINISH -> skipping connect_to_next_hop()")
                self.sock_next_hop = None
                return True

            if self._is_t3_target():
                logger.info(f"[CFG][{self.node_id}] next_hop=T3 (proxy via Manager). Skipping connect_to_next_hop().")
                self.sock_next_hop = None
                return True

            for client in self.clients:
                self._connect_to_next_hop(client)
                self.listen_for_previous_hop()
            return True
        return False

    def connect_manager_data(self):
        msg = self.recv_msg(self.manager_sock, "CLIENT_LIST")
        clients = msg[1]
        time.sleep(1)
        for c in range(len(clients)):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((helpers.dial_ip(self.proxies, self.ip, self.manager_ip), self.manager_port_data))
            msg = self.recv_msg(sock, "CONNECT_MANAGER_FORWARD_DATA")
            client_id = msg[1]
            self.manager_sock_data[client_id] = sock
            logger.info(f"Connected to Cluster Manager for client: {client_id}")


    def _get_local_model_shard(self, client_id):
        start = int(self.first_task['start_layer'])
        end = int(self.first_task['end_layer'])

        logger.info(f"Initializing Local Model Shard: Layers {start}-{end}")
        self.snapshot_start = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Snapshot BEFORE initialization:\n"
                    f"\t\t\t{self.snapshot_start}")

        is_last = self.local_finish

        new_model = ModelFactory.get_model_shard(
            model_name=self.model_name,
            model_type=self.model_type,
            start_layer=start,
            end_layer=end,
            is_first=False,
            is_last=is_last,
            device=self.device,
            hf_token=self.hf_token,
            num_classes=self.num_classes,
        )

        self.base_models_per_client[client_id] = new_model
        if self.model_type == "llm":
            new_optimizer = optim.AdamW(new_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        else:
            new_optimizer = optim.SGD(new_model.parameters(), lr=self.learning_rate, momentum=0.9)

        logger.info(f"Model Shard initialized successfully. is_last={is_last}")

        self.snapshot_end = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Snapshot AFTER initialization:\n"
                    f"\t\t\t{self.snapshot_end}")
        helpers.log_snapshots(node_id=self.node_id,
                              run_id=self.run_id,
                              snapshot_start=self.snapshot_start,
                              snapshot_end=self.snapshot_end,
                              description="Model_Shard_Load",
                              r=-1,
                              i=1)
        return new_model, new_optimizer

    def listen_for_previous_hop(self):
        """Accepts connection from Previous Node."""

        try:
            self.sock_server.bind((self.ip, self.local_port))
            self.sock_server.listen(5)
            logger.info(f"Listening for previous hop on {self.local_port}...")

            for c in self.clients:
                client_sock, _ = self.sock_server.accept()
                msg = self.recv_msg(client_sock,"PREVIOUS_HOP")  # Handshake
                if msg:
                    sender_id = msg[1]['previous_hop']
                    client_id = msg[1]['device_id']

                    logger.info(f"-> Accepted connection from: {sender_id}")
                    self.previous_hop = sender_id

                    if sender_id not in self.sock_previous_hop:
                        self.sock_previous_hop[sender_id] = {}

                    self.sock_previous_hop[sender_id][client_id] = client_sock


        except Exception as e:
            logger.error(f"Listener Error: {e}")

    def wait_start_training(self):
        logger.info(f"Waiting for START_TRAINING...")
        msg = self.recv_msg(self.manager_sock, "START_TRAINING")
        self.global_rounds = msg[1]["global_rounds"]
        self.iterations_train = msg[1]["iterations_train"]
        self.iterations_eval = msg[1]["iterations_eval"]
        self.learning_rate = msg[1]["learning_rate"]
        self.mu = msg[1].get("mu", 0.0)
        self.fedprox_cpu = msg[1].get("fedprox_cpu", False)
        self.weight_decay = msg[1].get("weight_decay", 0.01)
        self.alpha = msg[1]["alpha"]
        return True

    def start_training_as_worker(self):
        """Main Loop: Receives Data -> Forward -> Backward -> Update -> Eval."""
        logger.info("=== Training Mode Started ===")

        self.threads_training_clients = {}
        self.threads_evaluate_clients = {}
        self._save_experiment_configuration()


        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()
        # else:
        #     helpers.malloc_trim_if_possible()

        for r in range(self.global_rounds):
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info(f"\n--- Starting Round {r}/{self.global_rounds-1} ---")

            # --- LR Decay (once per round, before threads) ---
            if self.model_type == 'llm':
                if r > 0:
                    self.learning_rate *= 0.9
            elif self.model_type == 'vision':
                if r in (49, 74, 99, 124, 149):
                    self.learning_rate *= 0.1
            logger.info(f"[Round {r}] LR: {self.learning_rate}")

            # ---------------------------------------------------------
            # PHASE 1: TRAINING
            # ---------------------------------------------------------
            for device_id in self.clients:
                t = threading.Thread(
                    target=self._handle_chain_upstream,
                    args=(device_id, r),
                    name=f"T-Up-{device_id}",
                    daemon=True
                )
                self.threads_training_clients[device_id] = t
                t.start()

            for t in self.threads_training_clients.values():
                t.join()
            self._send_worker_train_done(round_idx=r)

            logger.info(f"[Round {r}] Training finished. Updating weights...")

            snapshot = helpers.get_local_resource_snapshot(self.device)
            logger.info(f"Snapshot: {snapshot}") ## It must be saved on file

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            self._sync_after_round(round_idx=r)
            # ---------------------------------------------------------
            # PHASE 3: EVALUATION
            # ---------------------------------------------------------
            # Wait for CM to tell us which clients to evaluate
            msg = self.recv_msg(self.manager_sock, "EVAL_CLIENTS")
            eval_clients = msg[1].get("clients", []) if msg else []

            if eval_clients:
                logger.info(f"[Round {r}] Starting Evaluation for {eval_clients}...")
                for device_id in eval_clients:
                    if device_id in self.clients:
                        t_eval = threading.Thread(
                            target=self.evaluate,
                            args=(device_id, r),
                            name=f"T-Eval-{device_id}",
                            daemon=True
                        )
                        self.threads_evaluate_clients[device_id] = t_eval
                        t_eval.start()

                for t in self.threads_evaluate_clients.values():
                    t.join()
            else:
                logger.info(f"[Round {r}] Evaluation skipped (no eval clients).")

            # --- RESOURCE READING BETWEEN ROUNDS ---

            self.answer_profile_request(round_idx=r)
            self.wait_reconfiguration(round_idx=r)

            logger.info(f"[Round {r}] Cycle Complete.")

    def _train(self, device_id, r):

        iterations_train = self.iterations_train[device_id]
        client_model, client_optimizer = self._get_client_model(device_id)
        client_model.train()


        # Apply current LR (already decayed in round loop)
        for param_group in client_optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        # FedProx: save global weights before local training
        if self.mu > 0:
            if self.fedprox_cpu:
                self.client_global_params[device_id] = {n: p.detach().cpu() for n, p in client_model.named_parameters()}
            else:
                self.client_global_params[device_id] = {n: p.clone().detach() for n, p in client_model.named_parameters()}

        self.snapshot_start = helpers.get_local_resource_snapshot(self.device)
        for i in tqdm.tqdm(range(iterations_train), desc=f"[{self.node_id} | {device_id} | Round: {r}]"):

            # msg = self._recv_forward_next_hop_for(device_id)

            msg = self.recv_msg(self.sock_previous_hop[self.previous_hop][device_id],
                                "FORWARD_DATA_NEXT_HOP")

            payload = msg[1]
            req_id = payload['req_id']

            input_raw = payload['data']
            labels_raw = payload['labels']
            mask_raw = payload.get('attention_mask', None)
            pos_raw = payload.get('position_ids', None)

            inputs = input_raw.to(self.device)
            labels = labels_raw.to(self.device)

            masks = mask_raw.to(self.device) if mask_raw is not None else None
            pos_ids = pos_raw.to(self.device) if pos_raw is not None else None

            if not inputs.requires_grad:
                inputs.requires_grad_(True)

            client_optimizer.zero_grad()

            t_start_comp = time.perf_counter()

            if self.model_type == 'llm':
                output = client_model(inputs,
                                      attention_mask=masks,
                                      position_ids=pos_ids)
            else:
                output = client_model(inputs)

            helpers.sync_device(self.device)

            t_end_comp = time.perf_counter()
            t_comp = t_end_comp - t_start_comp
            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name=f"T_comp_forward",
                duration=t_comp,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            self._save_context(req_id, inputs, output)

            t_start_d2h = time.perf_counter()

            finishes_locally = self.next_hop == "LOCAL_FINISH" or self.local_finish

            send_payload = {
                'req_id': req_id,
                'source': device_id,
                'data': None if finishes_locally else output.detach().cpu(),
                'original_source': payload.get('original_source', device_id),
                'labels': labels.cpu(),
                'attention_mask': masks.cpu() if masks is not None else None,
                'position_ids': pos_ids.cpu() if pos_ids is not None else None,
            }
            t_end_d2h = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_d2h_forward",
                duration=(t_end_d2h - t_start_d2h),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            helpers.log_payload_size(node_id=self.node_id, run_id=self.run_id,
                                     client_id=device_id, round_idx=r,
                                     hop="T2->next", payload=send_payload, batch_idx=i)

            self.snapshot_end = helpers.get_local_resource_snapshot(self.device)
            helpers.log_snapshots(node_id=self.node_id,
                                  run_id=self.run_id,
                                  snapshot_start=self.snapshot_start,
                                  snapshot_end=self.snapshot_end,
                                  r=r,
                                  i=i,
                                  description=f"During_train")

            self._handle_forward(send_payload,i,r)

        # FedProx: free global params snapshot after round
        self.client_global_params.pop(device_id, None)

    def evaluate(self, device_id, r):
        client_model, _ = self._get_client_model(device_id)
        client_model.eval()

        logger.info(f"[Eval {device_id}] Listening for evaluation stream...")

        iterations_eval = self.iterations_eval[device_id]
        for i in tqdm.tqdm(range(iterations_eval), desc=f"[{self.node_id} | {device_id} | Round: {r}]"):
            msg = self._recv_eval_for_device(device_id)
            if not msg:
                logger.error(f"[Eval {device_id}] prev-hop socket closed/invalid during eval stream.")
                break

            command = msg[0]

            if "FORWARD_EVAL" in command:
                payload = msg[1]

                input_gpu = payload['data'].to(self.device)
                mask_gpu = payload.get('attention_mask', None)
                pos_gpu = payload.get('position_ids', None)
                req_id = payload.get('req_id', None)

                mask_gpu = mask_gpu.to(self.device) if mask_gpu is not None else None
                pos_gpu = pos_gpu.to(self.device) if pos_gpu is not None else None
                t_start_eval = time.perf_counter()

                with torch.no_grad():
                    if self.model_type == 'llm':
                        output = client_model(input_gpu, attention_mask=mask_gpu, position_ids=pos_gpu)
                    else:
                        output = client_model(input_gpu)

                helpers.sync_device(self.device)

                t_end_eval = time.perf_counter()

                t_eval = (t_end_eval - t_start_eval)

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_eval",
                    duration=t_eval,
                    round_idx=-1,
                    batch_idx=-1,
                    req_id=req_id,
                    delay=None,
                )

                target_sock = self.manager_sock_data[device_id]
                if self._is_t3_target():
                    payload['data'] = output.detach().cpu()
                    self.send_msg(target_sock, ["FORWARD_EVAL_TO_TIER_3", payload])

                elif self.next_hop == "LOCAL_FINISH":
                    labels_gpu = payload.get('labels').to(self.device)
                    with torch.no_grad():
                        loss_sum, correct, n_tokens = helpers.compute_eval_batch_metrics(
                            model_type=self.model_type,
                            output=output,
                            labels=labels_gpu,
                            criterion=self.criterion,
                            device=self.device
                        )

                    metrics_payload = {
                        'req_id': req_id,
                        'loss_sum': loss_sum,
                        'correct': correct,
                        'n_tokens': n_tokens
                    }
                    self.send_msg(target_sock, ["EVAL_METRICS", metrics_payload])
                else:
                    payload['data'] = output.detach().cpu()
                    sock_next = self.sock_next_hop[self.next_hop][device_id]
                    self.send_msg(sock_next, ["FORWARD_EVAL_NEXT_HOP", payload])
            else:
                logger.debug(f"[Eval {device_id}] Ignoring unexpected cmd={command}")

        client_model.train()
        logger.info(f"[Eval {device_id}] Thread finished.")


    def _calculate_loss(self, output, labels):
        if self.model_type == 'vision':
            return self.criterion(output, labels)

        elif self.model_type == 'llm':
            logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            return self.criterion(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1)
            )
        return 0.0

    def _handle_chain_upstream(self, device_id,r):

            logger.info(f"[TRAIN {self.previous_hop}] Waiting for FORWARD_DATA_NEXT_HOP")

            try:
                self._train(device_id, r)
            except Exception as exc:
                helpers.report_fatal(
                    exc, node_id=self.node_id, run_id=self.run_id, seed=self.seed,
                    client_id=device_id, round_idx=r, batch_idx=-1,
                    device=self.device, phase="train",
                    layers="%s-%s" % (self.start_layer, self.end_layer))
            logger.info(f"[Train {device_id}] Round {r} finished")
            logger.info(f"[Train {device_id}] Waiting for aggregation...")

    def _handle_local_weights_request_tagged(self, msg, expected_round: int):
        payload = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
        req_round = payload.get("round", None)
        req_id = payload.get("req_id", "no_req_id")
        fp16 = bool(payload.get("fp16", False))
        client_ids = payload.get("client_ids", [])

        if req_round is not None and req_round != expected_round:
            logger.warning(
                f"[WEIGHTS][T2W] round mismatch (got={req_round}, expected={expected_round}) req_id={req_id}")

        weights_by_client = {}
        for cid in client_ids:
            if cid in self.models:
                weights_by_client[cid] = helpers.state_dict_cpu(self.models[cid], fp16=fp16)

        resp = {
            "round": expected_round,
            "req_id": req_id,
            "packs_by_client": {}
        }

        for cid in client_ids:
            if cid in self.models:
                sd = helpers.state_dict_cpu(self.models[cid], fp16=fp16)
                resp["packs_by_client"][cid] = [{
                    "shard": {"tier": "t2", "start": int(self.start_layer), "end": int(self.end_layer)},
                    "weights": sd
                }]
        with self.manager_send_lock:
            self.send_msg(self.manager_sock, ["LOCAL_WEIGHTS_RESPONSE_TAGGED", resp])

        logger.info(f"[WEIGHTS][T2W] sent weights for {len(weights_by_client)} clients fp16={fp16} req_id={req_id}")

    def _apply_local_update_tagged(self, msg, r):
        # CM sends: ["UPDATE_LOCAL_WEIGHTS_TAGGED", {"weights": new_weights, "client_id": X}]
        payload = msg[1]
        new_weights = payload["weights"]
        target_client = payload["client_id"]

        if target_client in self.models:


            self.models[target_client].load_state_dict(new_weights, strict=False)

            if self.model_type == "llm":
                # Keep optimizer state (Adam moments) — only refresh LR
                for pg in self.optimizers[target_client].param_groups:
                    pg['lr'] = self.learning_rate
            else:
                self.optimizers[target_client] = optim.SGD(self.models[target_client].parameters(),
                                                       lr=self.learning_rate,
                                                       momentum=0.9)

            logger.info(f"[Federated][T2W] Updated client={target_client}")


        else:
            logger.warning(f"[Federated][T2W] No model replica for client={target_client} (ignored)")

    def _sync_after_round(self, round_idx: int):
        while True:
            msg = self.recv_msg(self.manager_sock)
            if not msg:
                raise RuntimeError("[T2W] manager socket closed during sync")

            cmd = msg[0]

            if cmd == "REQUEST_LOCAL_WEIGHTS_TAGGED":
                self._handle_local_weights_request_tagged(msg, expected_round=round_idx)
                continue

            if cmd == "UPDATE_LOCAL_WEIGHTS_TAGGED":
                self._apply_local_update_tagged(msg, round_idx)
                continue

            # Sentinel: CM finished sending all updates for this round
            if cmd == "UPDATE_DONE":
                break

            logger.warning(f"[T2W] Unexpected cmd during sync: {cmd} (ignored)")

    def _save_context(self, req_id, input_tensor, output_tensor):
        with self.context_lock:
            self.execution_context[req_id] = {'input': input_tensor,
                                              'output': output_tensor}

    def _get_context(self, req_id):
        with self.context_lock:
            if req_id in self.execution_context:
                ctx = self.execution_context.pop(req_id)
                return ctx['input'], ctx['output']
            return None, None

    def _handle_forward(self, payload, i, r):
        """
        Sends forward data and BLOCKS waiting for the corresponding backward (synchronous).
        For LOCAL_FINISH (tail), computes loss locally and initiates backward.
        """
        req_id = payload['req_id']
        original_source = payload.get('original_source', payload.get('source', None))

        # =========================================================
        # CASE 0: CN is tail and is Strong Cluster
        # =========================================================
        if self.next_hop == "LOCAL_FINISH" or self.local_finish:

            client_model, client_optimizer = self._get_client_model(original_source)
            inputs, outputs = self._get_context(req_id)

            labels = payload['labels'].to(self.device)

            # client_optimizer.zero_grad()

            t_start_comp_loss = time.perf_counter()

            loss = self._calculate_loss(outputs, labels)
            loss_value = loss.item()
            loss.backward()

            # FedProx proximal gradient
            gp = self.client_global_params.get(original_source)
            if self.mu > 0 and gp is not None:
                for name, param in client_model.named_parameters():
                    if param.grad is not None and name in gp:
                        param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

            torch.nn.utils.clip_grad_norm_(client_model.parameters(),
                                           max_norm=self.clip_max_norm)
            client_optimizer.step()

            helpers.sync_device(self.device)

            t_end_comp_loss = time.perf_counter()
            t_comp_loss = t_end_comp_loss - t_start_comp_loss

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comp_backward",
                duration=t_comp_loss,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            metrics = {'loss': loss_value}

            if self.model_type == 'vision':
                metrics['feat_mean'] = inputs.mean().item()
                metrics['feat_std'] = inputs.std().item()

            group_id = 0

            helpers.log_training_metrics(
                run_id=self.run_id,
                client_id=original_source,
                group_id=group_id,
                round_idx=r,
                iteration_idx=i,
                model_type=self.model_type,
                metrics=metrics,
                alpha=self.alpha,
                seed=self.seed
            )


            # grad_out = inputs.grad.detach()

            t_start_d2h = time.perf_counter()

            back_payload = {
                'req_id': req_id,
                'grad': inputs.grad.detach().cpu(),
                'original_source': original_source,
            }

            t_end_d2h = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_d2h_backward",
                duration=(t_end_d2h - t_start_d2h),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            helpers.log_payload_size(node_id=self.node_id, run_id=self.run_id,
                                     client_id=original_source, round_idx=r,
                                     hop="bwd", payload=back_payload, batch_idx=i)
            # with self.manager_send_lock:
            t_start_comm_backward = time.perf_counter()

            self.send_msg(self.sock_previous_hop[self.previous_hop][original_source],
                          ["BACKWARD_DATA_FROM_PREVIOUS_NODE", back_payload])

            t_end_comm_backward = time.perf_counter()
            t_comm_backward = t_end_comm_backward - t_start_comm_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comm_backward",
                duration=t_comm_backward,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            return
        # =========================================================
        # CASE 1: CN is tail and is Straggler Cluster
        # =========================================================
        if self._is_t3_target():
            # print("CASO 1: CN is tail and is Straggler Cluster")

            # target_sock = self.manager_sock
            target_sock = self.manager_sock_data[original_source]
            # with self.manager_send_lock:
            # print(f"CASO 1 [{original_source}]: CN is tail and is Straggler Cluster")
            t_start_comm = time.perf_counter()

            self.send_msg(target_sock, ["FORWARD_DATA_TO_TIER_3", payload])

            t_end_comm = time.perf_counter()
            t_comm = t_end_comm - t_start_comm
            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comm_forward",
                duration=t_comm,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )


            t_start_wait = time.perf_counter()

            msg = self.recv_msg(target_sock, "BACKWARD_DATA_FROM_CLUSTER_MANAGER")

            t_end_wait = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_wait_chain",
                duration=(t_end_wait - t_start_wait),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            payload = msg[1]

            req_id = payload.get('req_id', None)
            original_source = payload.get('original_source', None)
            grad_tensor = payload['grad'].to(self.device)
            client_model, client_optimizer = self._get_client_model(original_source)

            inputs, outputs = self._get_context(req_id)

            t_start_comp_backward = time.perf_counter()

            grad_tensor = grad_tensor.to(dtype=outputs.dtype)
            outputs.backward(grad_tensor)

            # FedProx proximal gradient
            gp = self.client_global_params.get(original_source)
            if self.mu > 0 and gp is not None:
                for name, param in client_model.named_parameters():
                    if param.grad is not None and name in gp:
                        param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

            torch.nn.utils.clip_grad_norm_(client_model.parameters(),
                                           max_norm=self.clip_max_norm)

            client_optimizer.step()

            helpers.sync_device(self.device)

            t_end_comp_backward = time.perf_counter()
            t_comp_backward = t_end_comp_backward - t_start_comp_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comp_backward",
                duration=t_comp_backward,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            grad_out = inputs.grad.detach().cpu()

            t_start_d2h = time.perf_counter()

            back_payload = {
                'req_id': req_id,
                'grad': grad_out,
                'original_source': original_source
            }

            t_end_d2h = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_d2h_backward",
                duration=(t_end_d2h - t_start_d2h),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            helpers.log_payload_size(node_id=self.node_id, run_id=self.run_id,
                                     client_id=original_source, round_idx=r,
                                     hop="bwd", payload=back_payload, batch_idx=i)

            t_start_comm_backward = time.perf_counter()

            self.send_msg(self.sock_previous_hop[self.previous_hop][original_source],
                          ["BACKWARD_DATA_FROM_PREVIOUS_NODE", back_payload])

            t_end_comm_backward = time.perf_counter()
            t_comm_backward = t_end_comm_backward - t_start_comm_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comm_backward",
                duration=t_comm_backward,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            return

        # =========================================================
        # CASE 2: CN is not tail
        # =========================================================
        else:

            target_sock = self.sock_next_hop[self.next_hop][original_source]

            t_start_comm = time.perf_counter()

            self.send_msg(target_sock, ["FORWARD_DATA_NEXT_HOP", payload])

            t_end_comm = time.perf_counter()
            t_comm = t_end_comm - t_start_comm
            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comm_forward",
                duration=t_comm,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            t_start_wait = time.perf_counter()

            msg = self.recv_msg(target_sock, "BACKWARD_DATA_FROM_PREVIOUS_NODE")

            t_end_wait = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_wait_chain",
                duration=(t_end_wait - t_start_wait),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            payload = msg[1]

            req_id = payload.get('req_id', None)
            original_source = payload.get('original_source', None)
            grad_tensor = payload['grad'].to(self.device)
            client_model, client_optimizer = self._get_client_model(original_source)

            inputs, outputs = self._get_context(req_id)

            try:
                grad_tensor = grad_tensor.to(dtype=outputs.dtype)
            except Exception:
                pass

            t_start_comp_backward = time.perf_counter()

            outputs.backward(grad_tensor)

            # FedProx proximal gradient
            gp = self.client_global_params.get(original_source)
            if self.mu > 0 and gp is not None:
                for name, param in client_model.named_parameters():
                    if param.grad is not None and name in gp:
                        param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

            torch.nn.utils.clip_grad_norm_(client_model.parameters(),
                                           max_norm=self.clip_max_norm)

            client_optimizer.step()

            helpers.sync_device(self.device)

            t_end_comp_backward = time.perf_counter()
            t_comp_backward = t_end_comp_backward - t_start_comp_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comp_backward",
                duration=t_comp_backward,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            grad_out = inputs.grad.detach().cpu()

            t_start_d2h = time.perf_counter()

            back_payload = {
                'req_id': req_id,
                'grad': grad_out,
                'original_source': original_source
            }

            t_end_d2h = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_d2h_backward",
                duration=(t_end_d2h - t_start_d2h),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            helpers.log_payload_size(node_id=self.node_id, run_id=self.run_id,
                                     client_id=original_source, round_idx=r,
                                     hop="bwd", payload=back_payload, batch_idx=i)

            t_start_comm_backward = time.perf_counter()

            self.send_msg(self.sock_previous_hop[self.previous_hop][original_source],
                          ["BACKWARD_DATA_FROM_PREVIOUS_NODE", back_payload])

            t_end_comm_backward = time.perf_counter()
            t_comm_backward = t_end_comm_backward - t_start_comm_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comm_backward",
                duration=t_comm_backward,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            return

    def _recv_forward_next_hop_for(self, device_id: str):
        """
        Reads from the previous hop under a lock and delivers the FORWARD_DATA_NEXT_HOP
        message for this device, buffering any that belong to another.
        """
        while True:
            # 1) If there is a buffered message for this client, return it
            with self.prevhop_recv_lock:
                q = self.prevhop_inbox.get(device_id)
                if q:
                    return q.pop(0)

            # 2) Otherwise, read ONE message from the socket (lock to avoid race)
            # with self.prevhop_recv_lock:
            msg = self.recv_msg(self.sock_previous_hop[self.previous_hop][device_id])

            if not msg:
                return None

            cmd = msg[0]
            if cmd != "FORWARD_DATA_NEXT_HOP":
                continue

            payload = msg[1] if len(msg) > 1 else {}
            src = payload.get("source")

            # 3) If it belongs to the correct client, return it
            if src == device_id:
                return msg

            # 4) If it belongs to another client, buffer it for that client
            with self.prevhop_recv_lock:
                self.prevhop_inbox.setdefault(src, []).append(msg)

    def _recv_eval_for_device(self, device_id: str):
        """
        Receives messages from the previous hop and demultiplexes to the correct client.
        Updated to ignore EVAL_FINISHED and focus on FORWARD_EVAL.
        """
        while True:

            msg = self.recv_msg(self.sock_previous_hop[self.previous_hop][device_id])

            if not msg:
                return None

            cmd = msg[0]
            target = None

            # 3) Determine the target based on the data payload
            if "FORWARD_EVAL_NEXT_HOP" in str(cmd):
                payload = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
                # The target is the original source of the data (client_id)
                target = payload.get("source") or payload.get("original_source")


            else:
                continue

            # 4) Distribution
            if target == device_id:
                return msg

            if target is None:
                continue

    def wait_reconfiguration(self, round_idx):
        """
        Hears what the Manager decided for this round boundary. Exactly one message
        arrives, and when the range changed it carries the parameters of the new one.
        """
        msg = self.recv_msg(self.manager_sock)
        if not msg:
            logger.error(f"[REALLOC][{self.node_id}] the Manager's decision did not arrive.")
            return False

        if msg[0] != "RECONFIGURE_NODE":
            return False

        started = time.perf_counter()
        payload = msg[1] or {}
        task = payload.get("task") or {}
        weights = payload.get("weights") or {}
        old_range = (self.start_layer, self.end_layer)

        self.first_task = task
        self.start_layer = int(task['start_layer'])
        self.end_layer = int(task['end_layer'])
        self.should_offload = task.get('should_offload', self.should_offload)
        self.is_tail = task.get('is_tail', self.is_tail)
        self.local_finish = task.get('local_finish', self.local_finish)
        self.next_hop = task['next_hop']
        self.next_port = task['next_port']
        self.next_ip = task['next_ip']


        for client_id, model in self.models.items():
            optimizer = self.optimizers.get(client_id)
            if optimizer is None:
                continue
            held = helpers.capture_optimizer_moments(model, optimizer)
            kept, left = helpers.carry_optimizer_moments(
                held, old_range[0], self.start_layer, self.end_layer,
                prefix=helpers.shard_layer_prefix(self.model_type))
            self.pending_optimizer_moments[client_id] = kept
            logger.info(f"[REALLOC][{self.node_id}] {client_id}: carried the moments "
                        f"of {len(kept)} parameters over the new range "
                        f"{self.start_layer}-{self.end_layer}, {left} left behind "
                        f"with the layers that moved.")

        self.models = {}
        self.optimizers = {}
        if weights:
            for client_id in self.clients:
                self.pending_weight_packs[client_id] = weights
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - started
        helpers.log_performance_metric(
            node_id=self.node_id, client_id="ALL", seed=self.seed, run_id=self.run_id,
            metric_name="T_reallocation_apply", duration=elapsed, round_idx=round_idx,
            batch_idx=-1, req_id=None, delay=None)

        with self.manager_send_lock:
            self.send_msg(self.manager_sock,
                          ["RECONFIGURE_NODE_DONE", {"round": round_idx,
                                                     "node_id": self.node_id,
                                                     "t_apply_s": elapsed}])
        logger.info(f"[REALLOC][{self.node_id}] {old_range} -> "
                    f"({self.start_layer}, {self.end_layer}) | next_hop={self.next_hop} "
                    f"| local_finish={self.local_finish} | {elapsed:.2f}s")
        return True

    def answer_profile_request(self, round_idx):
        """
        Measures this device again between rounds, at the Manager's request, with the
        shard resident and nothing in flight.
        """
        msg = self.recv_msg(self.manager_sock, "REQUEST_NODE_PROFILE")
        if not msg:
            logger.error(f"[HW][{self.node_id}] profile request not received.")
            return False


        reading = helpers.measure_local_resources(self.node_id, self.device, runtime=True)
        reading["round"] = round_idx

        with self.manager_send_lock:
            self.send_msg(self.manager_sock, ["NODE_PROFILE_REPORT", reading])
        return True

    def _send_worker_train_done(self, round_idx: int):
        try:
            payload = {"round": round_idx, "worker_id": self.node_id}
            with self.manager_send_lock:
                self.send_msg(self.manager_sock, ["WORKER_TRAIN_DONE", payload])
                logger.info(f"[SYNC][{self.node_id}] Sent WORKER_TRAIN_DONE | round={round_idx}")
        except Exception as e:
            logger.error(f"[SYNC][{self.node_id}] Failed to send WORKER_TRAIN_DONE: {e}")

    def _is_t3_target(self):

        # `_t3_` is the Aggregation Pool tag in the node ids (e.g. s1_t3_1)
        return (
            self.next_hop == "tier_3_target"
            or (self.next_port == self.orchestrator_ports[1])
            or ("_t3_" in str(self.next_hop).lower())
        )

    def _save_experiment_configuration(self):
        """
        Collects and persists all metadata for the current experiment.
        Includes: Hyperparameters, Topology, Hardware, Splits, and Strong/Straggler classification.
        """
        logger.info("=== Saving Experiment Configuration Snapshot ===")
        metadata = {
            "meta": {
                "run_id": self.run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "seed": self.seed,
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device": self.my_hw,
            }
        }
        helpers.save_experiment_metadata(self.node_id, self.run_id, metadata)