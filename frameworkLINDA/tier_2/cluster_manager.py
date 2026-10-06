# @author: Marcio Lopes
import copy
import gc
import random
import socket
import sys
import time
import datetime
import psutil
import torch.nn as nn
import transformers
import numpy as np
import torch
import json
import threading
import torch.optim as optim
import tqdm
from sympy.codegen.cnodes import sizeof

sys.path.append('../../')
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers
from utils.standardTuples.data_payload import ForwardDataTuple, BackwardGradTuple


class ClusterManager(CommunicationModule):
    """
    Tier 2 Component: Manages a Local Pool of Resources.
    It must run on "first" GPU Server.
    """

    def __init__(self,
                 node_id,
                 site_name,
                 ip_address,
                 local_port_list,
                 orchestrator_ip,
                 orchestrator_ports,
                 topology,
                 device):
        """
        Args:
            node_id: ID of this site (e.g., 's3_t2_3')
            ip_address: Local IP of this manager (for Tier 1 to connect later)
            orchestrator_ip: IP of Tier 3 Leader
            orchestrator_ports: Port of Tier 3 Leader
        """
        super().__init__(node_id=node_id, ip_address=ip_address)  # ip_address)

        self.weight_decay = None
        self.fedprox_cpu = None
        self.check = True
        self.all_clients = []
        self.worker_data_forward = {}
        self.clip_max_norm = None
        self.learning_rate = None
        self.tier1_devices_data = {}
        self.next_ip = None
        self.next_port = None
        self.next_hop = None
        self.threads_evaluate_clients = None
        self.base_models_per_client = {}
        self.client_train_sockets = {}
        self.context_timestamps = []
        self.first_task = None
        self.site_name = site_name
        self.ip_address = ip_address
        self.local_port_list = local_port_list
        self.port_compute = local_port_list[0]
        self.port_devices_control = local_port_list[1]
        self.port_devices_data = local_port_list[2]
        self.port_worker_data = local_port_list[3]
        self.workers_hardware = {}
        self.site_profiles = {}
        self.chain_in_force = []
        self.chain_counts_in_force = {}
        self.pending_worker_tasks = {}
        self.pending_weight_packs = {}
        self.pending_optimizer_moments = {}
        self.worker_task_ranges = {}
        self.worker_chain_locks = {}
        self.active_workers = set()
        self.delay = 0.0
        self.snapshot_start = {}
        self.snapshot_end = {}

        self.orchestrator_ip = orchestrator_ip

        self.orchestrator_ports = orchestrator_ports
        self.topology = topology[0] if isinstance(topology, tuple) else topology
        self.proxies = helpers.load_proxy_map(self.topology)

        self.orchestrator_sock = None
        self.worker_data_sockets = {}
        self.eval_chain_locks = {}
        self.tier1_devices_control = {}
        self.local_compute_nodes = {}
        self.global_rounds = None
        self.current_allocation = {}
        self.client_id_map = {}
        self.local_tasks = {}
        self.last_node = {}
        self.models = {}
        self.base_model =None
        self.optimizers = {}

        if device == "cpu":
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                gpu_id = int(device.split(":")[1])
                torch.cuda.set_device(gpu_id)
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")


        self.execution_context = {}
        self.context_lock = threading.Lock()

        self.train_lock = threading.Lock()
        self.batch_size = None
        self.model_name = None
        self.model_type = None
        self.hf_token = HF_TOKEN
        self.num_classes = None
        self.total_clients = 0
        self.should_offload = None
        self.seed = None
        self.alpha = 0.5
        self.iterations_train = {}
        self.iterations_eval = {}
        self.client_norms = {}

        self.orchestrator_send_lock = threading.Lock()
        self.threads_training_clients = {}
        self.criterion = None

        self.run_id = None
        self.my_hw = helpers.get_local_device_name(self.device)

        logger.info(f"Cluster Manager {node_id} initialized at {ip_address}")
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

    def connect_to_orchestrator_control(self):
        """Establish connection with Tier 3 Leader Control Plane."""
        control_port = self.orchestrator_ports[0]
        dial = helpers.dial_ip(self.proxies, self.ip, self.orchestrator_ip)
        logger.info(f"Attempting Control Connection to {self.orchestrator_ip}:{control_port} "
                    f"(dialing {dial}:{control_port})...")

        try:
            self.sock_client.connect((dial, control_port))
            self.orchestrator_sock = self.sock_client
            msg = ["HELLO_TIER2", self.site_name]

            self.send_msg(self.orchestrator_sock, msg)
            logger.info(f"-> Control Plane Established (Port {control_port}).")
            self.perform_time_sync_handshake(self.orchestrator_sock, role="client")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Orchestrator Control: {e}")
            return False

    def connect_to_orchestrator_data(self):
        """Establish connection with Tier 3 Leader Data Plane."""
        data_port = self.orchestrator_ports[1]
        dial = helpers.dial_ip(self.proxies, self.ip, self.orchestrator_ip)
        logger.info(f"Attempting Data Connection to {self.orchestrator_ip}:{data_port} "
                    f"(dialing {dial}:{data_port})...")

        try:
            self.orchestrator_data_sock.connect((dial, data_port))
            logger.info(f"[DATA CONNECTION] {self.orchestrator_data_sock.getpeername()} AND {self.orchestrator_data_sock.getsockname()}.")
            msg_data = ["HELLO_DATA", self.node_id]
            self.send_msg(self.orchestrator_data_sock, msg_data)
            logger.info(f"-> [{msg_data[0]}] Data Plane: Trying establishing connection with "
                        f"{self.orchestrator_ip} (Port {data_port}).")



            return True

        except Exception as e:
            logger.error(f"Failed to connect to Orchestrator Data: {e}")
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

        try:
            logger.info(f"Connecting to Next Hop ([{target_id}][{device_id}]): {self.next_ip}:{target_port} "
                        f"(dialing {target_ip}:{target_port})...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((target_ip, target_port))

            if target_port != self.orchestrator_ports[1]:
                # msg = ["PREVIOUS_HOP", self.node_id]
                msg = ["PREVIOUS_HOP", {"previous_hop": self.node_id, "device_id": device_id}]
                self.send_msg(sock, msg)
                # self.sock_next_hop[self.next_hop][device_id] = sock
                logger.info("-> Connected to Next Hop.")
            return sock
        except Exception as e:
            logger.error(f"Failed to connect next hop: {e}")
            return False

    def _establish_dedicated_data_channels(self):
        """Creates a dedicated TCP connection for each local Tier 1 client to the Orchestrator."""
        logger.info("Establishing dedicated Data Channels for local clients...")

        for client_id, dev_info in self.tier1_devices_control.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                dial = helpers.dial_ip(self.proxies, self.ip, self.orchestrator_ip)
                sock.connect((dial, self.orchestrator_ports[1]))

                self.send_msg(sock, ["HELLO_CLIENT_TRAIN", client_id])

                self.client_train_sockets[client_id] = sock
                logger.info(f"   -> Dedicated Channel created for {client_id}")

            except Exception as e:
                logger.error(f"Failed to create channel for {client_id}: {e}")

    def _get_local_resource_snapshot(self):
        """
        Collects real-time GPU/CPU telemetry.
        Returns a dict with VRAM (if GPU) or RAM (if CPU) in GB.
        """
        resources = {
            "device": str(self.device),
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_free_gb": 0.0
        }

        if self.device.type == 'cuda':
            try:
                # Returns (free, total) in bytes
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)

                resources["vram_free_gb"] = free_bytes / (1024 ** 3)
                resources["vram_total_gb"] = total_bytes / (1024 ** 3)

                resources["gpu_name"] = torch.cuda.get_device_name(self.device)

            except Exception as e:
                logger.error(f"Error reading VRAM: {e}")

        else:
            try:
                mem = psutil.virtual_memory()
                resources["ram_total_gb"] = mem.total / (1024 ** 3)
                resources["ram_free_gb"] = mem.available / (1024 ** 3)
            except Exception as e:
                logger.error(f"Error reading RAM (psutil): {e}")

        return resources

    def connect_device_data(self,device_id):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.ip, self.port_devices_data))
        sock.listen(1)
        client_sock, (ip, port) = sock.accept()
        msg = self.recv_msg(client_sock, "HELLO_DEVICE_DATA")
        self.tier1_devices_data[device_id] = client_sock
        self.all_clients.append(device_id)

    def start_server_for_devices(self):
        """
        Runs in a separate thread. Listens for Tier 1 connections.
        """
        expected_devices = len(self.current_allocation)

        if expected_devices == 0:
            logger.info("No Tier 1 devices allocated to this cluster. Skipping listener.")
            return

        if not self.port_devices_control:
            logger.error("No port configured for devices (Index 1 missing)!")
            return

        logger.info(f"[DevServer] Waiting for {expected_devices} Devices on {self.ip}:{self.port_devices_control}...")

        # try:
        self.sock_manager_to_device.bind((self.ip, self.port_devices_control))
        self.sock_manager_to_device.listen(expected_devices + 2)

        while len(self.tier1_devices_control) < expected_devices:
            client_sock, (ip, port) = self.sock_manager_to_device.accept()

            msg = self.recv_msg(client_sock, "HELLO_DEVICE")
            if not msg: continue
            payload = msg[1]

            if isinstance(payload, list) and len(payload) == 3:
                device_id, hw_name, measured = payload
                self.workers_hardware[device_id] = hw_name
                self.site_profiles[device_id] = measured
            elif isinstance(payload, list) and len(payload) == 2:
                device_id, hw_name = payload
                self.workers_hardware[device_id] = hw_name
            else:
                device_id = payload
                self.workers_hardware[device_id] = "Unknown (Old Protocol)"

            self.perform_time_sync_handshake(client_sock, role="server")

            self.connect_device_data(device_id)

            if device_id in self.current_allocation:
                self.tier1_devices_control[device_id] = client_sock
                logger.info(f"[DevServer] Device Connected: {device_id}")

                alloc_line = self.current_allocation[device_id]
                client_training_id = self.client_id_map.get(device_id, -1)

                config_payload = {
                    'split_point': alloc_line.tier1_split_point,
                    'client_training_id': client_training_id,
                    'total_clients':self.total_clients,
                    'seed': self.seed,
                    'alpha': self.alpha,
                    'batch_size': self.batch_size,
                    'model_name': self.model_name,
                    'model_type': self.model_type,
                    'run_id': self.run_id,
                    'learning_rate': self.learning_rate
                }
                self.send_msg(client_sock, ["AGENT_CONFIG", config_payload])
                msg = self.recv_msg(client_sock, "ITERATIONS")
                self.iterations_train[device_id]=msg[1]
                self.iterations_eval[device_id]=msg[2]
                logger.info(f"Device: {device_id} Iteration: {self.iterations_train[device_id]}")
                logger.info(f"-> [DevServer] Config sent to {device_id}")
            else:
                logger.warning(f"-> [DevServer] Device {device_id} rejected.")
                client_sock.close()
        self._establish_dedicated_data_channels()
        logger.info("[DevServer] All expected devices connected!")
        # except Exception as e:
        #     logger.error(f"[DevServer] Error: {e}")

    def collect_runtime_profiles(self, round_idx):
        """
        Gathers the site's resource state between rounds and reports it upward as one
        report for the whole site. Called after the evaluation of the round.
        """
        msg = self.recv_msg(self.orchestrator_sock, "REQUEST_SITE_PROFILE")
        if not msg:
            logger.error("[HW] runtime profile request not received from the Orchestrator.")
            return False

        arrived_synced = self.get_synced_time()
        started = time.perf_counter()
        profiles = {}

        for worker_id, sock in self.local_compute_nodes.items():
            try:
                self.send_msg(sock, ["REQUEST_NODE_PROFILE", {"round": round_idx}])
                answer = self.recv_msg(sock, "NODE_PROFILE_REPORT")
                if answer:
                    profiles[worker_id] = answer[1]
                else:
                    logger.warning(f"[HW] {worker_id} did not answer the profile request.")
            except Exception as e:
                logger.error(f"[HW] could not read {worker_id}: {e}")

        profiles[self.node_id] = helpers.measure_local_resources(
            self.node_id, self.device, runtime=True)
        profiles[self.node_id]["round"] = round_idx

        collect_s = time.perf_counter() - started
        self.send_msg(self.orchestrator_sock, ["SITE_PROFILE_REPORT", {
            "site": self.site_name,
            "round": round_idx,
            "profiles": profiles,
            "t_arrived_synced": arrived_synced,
            "t_collect_s": collect_s,
        }])
        logger.info(f"[HW] round {round_idx}: reported {len(profiles)} readings for "
                    f"{self.site_name} in {collect_s:.2f}s")
        return True

    @staticmethod
    def _node_ranges_from_counts(order, counts, t1_split, should_offload):
        """
        Returns (start, end, terminates) per node, from the block counts. A node can keep
        its count and still hold different layers, because a node before it changed.
        """
        actives = [node for node in order if int(counts.get(node, 0)) > 0]
        out, cursor = {}, int(t1_split)
        for node in actives:
            blocks = int(counts[node])
            out[node] = (cursor, cursor + blocks,
                         bool(node == actives[-1] and not should_offload))
            cursor += blocks
        return out

    def wait_allocation_decision(self, round_idx):
        """
        Hears what the Orchestrator decided for this round boundary and passes it down the
        chain. One message always arrives, whether or not anything changed.
        """
        msg = self.recv_msg(self.orchestrator_sock)
        if not msg:
            logger.error("[REALLOC][CM] the Orchestrator's decision did not arrive.")
            return False

        if msg[0] == "UPDATE_ALLOCATION":
            return self.apply_new_allocation(round_idx, msg[1] or {})

        for node, sock in self.local_compute_nodes.items():
            self.send_msg(sock, ["KEEP_NODE_CONFIG", {"round": round_idx}])
        return False

    def apply_new_allocation(self, round_idx, spec):
        """
        Adopts a placement the Orchestrator re-planned and hands it down the chain, with
        the parameters of each new range. Returns False when nothing was applied.
        """
        started = time.perf_counter()
        chain = [(node, int(blocks)) for node, blocks in spec.get("chain", [])]
        ranges = {node: (int(a), int(b)) for node, a, b in spec.get("ranges", [])}
        packs = spec.get("packs", {}) or {}
        should_offload = bool(spec.get("should_offload", self.should_offload))
        t1_split = int(spec.get("t1_split_point", 1))

        incoming = [node for node, _ in chain]
        current = [node for node in self.chain_in_force if node in self.chain_in_force]
        if current and incoming != current:
            logger.error(f"[REALLOC][CM] the plan changes the chain "
                         f"({current} -> {incoming}); refused.")
            for node, sock in self.local_compute_nodes.items():
                self.send_msg(sock, ["KEEP_NODE_CONFIG", {"round": round_idx}])
            self.send_msg(self.orchestrator_sock,
                          ["UPDATE_ALLOCATION_DONE", {"round": round_idx,
                                                      "applied": False,
                                                      "reason": "membership_change"}])
            return False

        logger.info(f"[REALLOC][CM] round {round_idx}: {self.chain_counts_in_force} "
                    f"-> {dict(chain)} | offload {self.should_offload} -> {should_offload}")

        before = self._node_ranges_from_counts(
            self.chain_in_force, self.chain_counts_in_force, t1_split, self.should_offload)
        after = self._node_ranges_from_counts(incoming, dict(chain), t1_split, should_offload)
        affected = {node for node in after if after[node] != before.get(node)}
        logger.info(f"[REALLOC][CM] affected by the new plan: {sorted(affected)}")

        self.should_offload = should_offload

        for client_id, split_tuple in self.current_allocation.items():
            split_tuple.tier2_distribution.clear()
            split_tuple.tier2_distribution.update((node, blocks) for node, blocks in chain)

            for index, (node, blocks) in enumerate(chain):
                start, end = ranges[node]
                is_tail = (index == len(chain) - 1)
                if not is_tail:
                    next_hop = chain[index + 1][0]
                    next_port = helpers.resolve_node_port(self.topology, self.site_name, next_hop)
                    next_ip = helpers.resolve_node_ip(self.topology, self.site_name, next_hop)
                elif should_offload:
                    next_hop = split_tuple.tier3_target
                    next_port = self.orchestrator_ports[1]
                    next_ip = self.orchestrator_ip
                else:
                    next_hop, next_port, next_ip = "LOCAL_FINISH", None, None

                task = {
                    'seed': self.seed, 'start_layer': start, 'end_layer': end,
                    'next_hop': next_hop, 'next_port': next_port, 'next_ip': next_ip,
                    'devices': list(self.current_allocation.keys()),
                    'model_name': self.model_name, 'model_type': self.model_type,
                    'iterations_train': self.iterations_train[client_id],
                    'iterations_eval': self.iterations_eval[client_id],
                    'should_offload': should_offload, 'is_tail': is_tail,
                    'local_finish': (not should_offload) and is_tail,
                    'run_id': self.run_id, 'num_classes': self.num_classes,
                }

                if node == self.node_id:
                    self.local_tasks[client_id] = task
                elif node in self.local_compute_nodes:
                    self.worker_task_ranges.setdefault(node, {})[client_id] = (start, end)
                    self.pending_worker_tasks.setdefault(node, task)

            self.last_node[client_id] = chain[-1][0] if chain else None

        if self.local_tasks:
            self.first_task = list(self.local_tasks.values())[0]

        for node, sock in self.local_compute_nodes.items():
            if node in affected and node in self.pending_worker_tasks:
                self.send_msg(sock, ["RECONFIGURE_NODE", {
                    "round": round_idx,
                    "task": self.pending_worker_tasks[node],
                    "weights": packs.get(node, {})}])
                if not self.recv_msg(sock, "RECONFIGURE_NODE_DONE"):
                    logger.error(f"[REALLOC][CM] {node} did not confirm its new range.")
            else:
                self.send_msg(sock, ["KEEP_NODE_CONFIG", {"round": round_idx}])
        self.pending_worker_tasks = {}

        if self.node_id in affected:

            old_range = before.get(self.node_id)
            new_range = after.get(self.node_id)
            if old_range and new_range:
                for client_id, model in self.models.items():
                    optimizer = self.optimizers.get(client_id)
                    if optimizer is None:
                        continue
                    held = helpers.capture_optimizer_moments(model, optimizer)
                    kept, left = helpers.carry_optimizer_moments(
                        held, old_range[0], new_range[0], new_range[1],
                        prefix=helpers.shard_layer_prefix(self.model_type))
                    self.pending_optimizer_moments[client_id] = kept
                    logger.info(f"[REALLOC][CM] {client_id}: carried the moments of "
                                f"{len(kept)} parameters over the new range "
                                f"{new_range[0]}-{new_range[1]}, {left} left behind "
                                f"with the layers that moved.")

            self.models = {}
            self.optimizers = {}
            own_pack = packs.get(self.node_id)
            if own_pack:
                for client_id in self.current_allocation:
                    self.pending_weight_packs[client_id] = own_pack
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        else:
            logger.info(f"[REALLOC][CM] {self.node_id} keeps its own range "
                        f"{before.get(self.node_id)}; its replicas stand.")

        self.chain_in_force = incoming
        self.chain_counts_in_force = dict(chain)

        elapsed = time.perf_counter() - started
        helpers.log_performance_metric(
            node_id=self.node_id, client_id="ALL", seed=self.seed, run_id=self.run_id,
            metric_name="T_reallocation_apply", duration=elapsed, round_idx=round_idx,
            batch_idx=-1, req_id=None, delay=None)

        self.send_msg(self.orchestrator_sock,
                      ["UPDATE_ALLOCATION_DONE", {"round": round_idx, "applied": True,
                                                  "t_apply_s": elapsed}])
        logger.info(f"[REALLOC][CM] applied in {elapsed:.2f}s")
        return True

    def report_site_profile(self):
        """
        Answers Tier 3's resource discovery with what every node of this site measured,
        before the allocation comes down. Tier 1 is not in this report.
        """
        msg = self.recv_msg(self.orchestrator_sock, "REQUEST_SITE_PROFILE")
        if not msg:
            logger.error("[HW] no profile request received from the Orchestrator.")
            return False

        self.site_profiles[self.node_id] = helpers.measure_local_resources(
            self.node_id, self.device)

        self.send_msg(self.orchestrator_sock, ["SITE_PROFILE_REPORT", {
            "site": self.site_name,
            "profiles": dict(self.site_profiles),
        }])
        logger.info(f"[HW] reported {len(self.site_profiles)} measured profiles "
                    f"for {self.site_name}")
        return True

    def wait_for_initial_allocation(self):
        """Blocking call. Waits for Allocation + Metadata."""
        logger.info("Waiting for Initial Allocation from Orchestrator...")
        msg = self.recv_msg(self.orchestrator_sock, "INITIAL_ALLOCATION")

        if msg:
            payload = msg[1]
            self.current_allocation = payload['allocation']
            self.client_id_map = payload['client_map']
            self.total_clients = payload['total_clients']
            self.should_offload = payload['should_offload']
            self.seed = payload['seed']
            self.alpha = payload['alpha']
            self.batch_size = payload['batch_size']
            self.model_name = payload['model_name']
            self.model_type = payload['model_type']
            self.num_classes = payload.get('num_classes', 10)
            self.run_id = payload['run_id']
            self.enable_net_metrics(self.run_id)
            self.learning_rate = payload['learning_rate']
            self._set_global_seed()

            if self.model_type == "llm":
                self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
                self.clip_max_norm = 1.0
            else:
                self.criterion = nn.CrossEntropyLoss()
                self.clip_max_norm = 1.0

            logger.info(f"-> Allocation Received! Managing {len(self.current_allocation)} devices.")
            logger.info(f"Sending config to Orchestrator... {self.orchestrator_sock}")
            self.send_msg(self.orchestrator_sock, "INITIAL_ALLOCATION_COMPLETE")
            return True
        return False

    def wait_for_local_workers(self, expected_workers):
        """Opens local server and waits for Secondary Compute Nodes."""
        logger.info(f"Waiting for {expected_workers} Local Compute Nodes on {self.ip}:{self.port_compute}...")
        try:
            try:
                self.sock_server.bind((self.ip, self.port_compute))
                self.sock_server.listen(5)
            except OSError:
                logger.warning("Socket already bound (reusing).")

            while len(self.local_compute_nodes) < expected_workers:
                client_sock, (ip, port) = self.sock_server.accept()
                msg = self.recv_msg(client_sock, "HELLO_COMPUTE")
                if msg:
                    self.perform_time_sync_handshake(client_sock, role="server")

                    payload = msg[1]
                    if isinstance(payload, list) and len(payload) == 3:
                        worker_id, hw_name, measured = payload
                        self.workers_hardware[worker_id] = hw_name
                        self.site_profiles[worker_id] = measured
                    elif isinstance(payload, list) and len(payload) == 2:
                        worker_id, hw_name = payload
                        self.workers_hardware[worker_id] = hw_name
                    else:
                        worker_id = payload
                        self.workers_hardware[worker_id] = "Unknown"

                    self.local_compute_nodes[worker_id] = client_sock
                    logger.info(f"Local Worker Connected: {worker_id}")

            return True
        except Exception as e:
            logger.error(f"Error waiting for workers: {e}")
            raise e

    def distribute_intra_cluster_tasks(self):
        """Calculates roles and distributes tasks."""
        logger.info("Distributing tasks to Local Compute Nodes...")

        workers_payload = {w_id: [] for w_id in self.local_compute_nodes.keys()}
        node_id = 0
        for device_id, split_tuple in self.current_allocation.items():
            chain = split_tuple.tier2_distribution or {}

            # ==========================================================
            # Filter out nodes that would produce empty shards (layer_count <= 0)
            # ==========================================================
            filtered_items = []
            for node_id, layer_count in chain.items():
                try:
                    lc = int(layer_count)
                except Exception:
                    lc = 0

                if lc > 0:
                    filtered_items.append((node_id, lc))
                else:
                    logger.warning(
                        f"[SHARD-EMPTY-SKIP] device={device_id} node={node_id} "
                        f"layer_count={layer_count} -> node removed from chain"
                    )

            chain_nodes = [nid for nid, _ in filtered_items]
            chain_counts = {nid: lc for nid, lc in filtered_items}

            current_start_layer = int(split_tuple.tier1_split_point)


            if not chain_nodes:
                self.last_node[device_id] = None
                logger.warning(
                    f"[TASK-SKIP] device={device_id}: tier2_distribution has no layers (>0). "
                    f"Tier 2 will be skipped for this device."
                )
                self._log_compact_chain(
                    device_id=device_id,
                    split_tuple=split_tuple,
                    chain_nodes=[],
                    chain_counts={},
                    tier2_ignored=True
                )
                continue

            self.last_node[device_id] = chain_nodes[-1]
            self.chain_in_force = list(chain_nodes)
            self.chain_counts_in_force = dict(chain_counts)

            if len(chain_nodes) > 1:
                self.next_hop = chain_nodes[1]
                self.next_port = helpers.resolve_node_port(self.topology, self.site_name, self.next_hop)
                self.next_ip = helpers.resolve_node_ip(self.topology, self.site_name, self.next_hop)
            else:
                self.next_hop = None
                self.next_port = None
                self.next_ip = None


            # ==========================================================
            # Build tasks on top of the FILTERED chain (no empty shards)
            # ==========================================================
            for i, node_id in enumerate(chain_nodes):
                layer_count = chain_counts[node_id]
                end_layer = current_start_layer + layer_count

                # ======================================================
                # Safety check: never generate start==end (or end<start)
                # ======================================================
                if end_layer <= current_start_layer:
                    logger.warning(
                        f"[SHARD-EMPTY-SKIP] device={device_id} node={node_id} "
                        f"invalid_range={current_start_layer}-{end_layer} -> node skipped"
                    )
                    continue
                # ======================================================
                # next_hop computed on filtered chain (skips empty nodes)
                # ======================================================
                if i < len(chain_nodes) - 1:
                    next_hop = chain_nodes[i + 1]
                    next_port = helpers.resolve_node_port(self.topology, self.site_name, next_hop)
                    next_ip = helpers.resolve_node_ip(self.topology, self.site_name, next_hop)

                else:
                    # LAST NODE IN THE CHAIN (TAIL)
                    if self.should_offload:
                        next_hop = split_tuple.tier3_target
                        next_port = self.orchestrator_ports[1]
                        next_ip = self.orchestrator_ip

                    else:
                        # Tail finishes locally (does not forward to T3)
                        next_hop = "LOCAL_FINISH"
                        next_port = None
                        next_ip = None

                is_tail = (node_id == chain_nodes[-1])
                local_finish = (not self.should_offload) and is_tail

                task = {
                    'seed': self.seed,
                    'start_layer': current_start_layer,
                    'end_layer': end_layer,
                    'next_hop': next_hop,
                    'next_port': next_port,
                    'next_ip': next_ip,
                    'devices': list(self.current_allocation.keys()),
                    'model_name': self.model_name,
                    'model_type': self.model_type,
                    'iterations_train': self.iterations_train[device_id],
                    'iterations_eval': self.iterations_eval[device_id],
                    'should_offload': self.should_offload,
                    'is_tail': is_tail,
                    'local_finish': local_finish,
                    'run_id': self.run_id,
                    'num_classes': self.num_classes,
                }

                if node_id == self.node_id:
                    self.local_tasks[device_id] = task
                    logger.info(f"[SELF-TASK] device={device_id} Layers {current_start_layer}-{end_layer}")


                elif node_id in self.local_compute_nodes:
                    if node_id not in self.worker_task_ranges:
                        self.worker_task_ranges[node_id] = {}
                    self.worker_task_ranges[node_id][device_id] = (int(current_start_layer), int(end_layer))
                    workers_payload[node_id].append(task)


                else:
                    logger.error(
                        f"[TASK-ERROR] device={device_id}: node_id={node_id} not found in local_compute_nodes "
                        f"and is not the manager itself. Check your topology/split_matrix."
                    )

                current_start_layer = end_layer
        # ==========================================================
        # Do not send empty COMPUTE_CONFIG (worker would stall)
        # ==========================================================
        for worker_id, tasks in workers_payload.items():
            if worker_id not in self.local_compute_nodes:
                continue

            if not tasks:
                logger.info(f"[TASK-SKIP] worker={worker_id}: no tasks after filtering empty shards. Not sending.")
                continue

            sock = self.local_compute_nodes[worker_id]
            logger.info(f"Sending TASKS to {worker_id}: {len(tasks)} task(s)")
            self.send_msg(sock, ["COMPUTE_CONFIG", tasks])
            logger.info(f"Configurations sent to {worker_id}")
            time.sleep(5.0)

            for dev_id, _ in self.current_allocation.items():
                new_sock = self._connect_to_next_hop(dev_id)
                # if new_sock == False:
                #     time.sleep(1.0)
                #     new_sock = self._connect_to_next_hop(dev_id)

                if self.next_hop not in self.worker_data_sockets:
                    self.worker_data_sockets[self.next_hop] = {}
                self.worker_data_sockets[self.next_hop][dev_id] = new_sock

        self.active_workers = {wid for wid, tasks in workers_payload.items() if tasks}
        logger.info(f"[CM] Active workers for training: {sorted(self.active_workers)}")

        if self.local_tasks:
            self.first_task = list(self.local_tasks.values())[0]

        logger.info("Intra-cluster task distribution completed.")

    def listen_worker_data(self):

        logger.info(f"Listening from Worker data connection for all clients")
        for worker_id in self.local_compute_nodes.keys():

            self.send_msg(self.local_compute_nodes[worker_id], ["CLIENT_LIST", self.all_clients])
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.ip, self.port_worker_data))
            for client_id in self.all_clients:
                sock.listen(len(self.all_clients)+1)
                worker_sock, (ip, port) = sock.accept()
                self.send_msg(worker_sock, ["CONNECT_MANAGER_FORWARD_DATA", client_id])
                if worker_id not in self.worker_data_forward:
                    self.worker_data_forward[worker_id] = {}
                self.worker_data_forward[worker_id][client_id] = worker_sock

                logger.info(f"Data client {client_id} in worker {worker_id} connected!")

    def _wait_workers_train_done(self, round_idx: int):
        """
        Explicit barrier: ensures all active workers finished the round
        before starting sync/aggregation (REQUEST_LOCAL_WEIGHTS_TAGGED / UPDATE_*).
        """
        if not self.active_workers:
            logger.info(f"[CM] No active workers. Skipping WORKER_TRAIN_DONE barrier. round={round_idx}")
            return

        logger.info(f"[CM] Waiting WORKER_TRAIN_DONE from {len(self.active_workers)} worker(s) | round={round_idx}")

        for wid in sorted(self.active_workers):
            sock = self.local_compute_nodes.get(wid)
            if sock is None:
                raise RuntimeError(f"[CM] active worker {wid} not found in local_compute_nodes")

            msg = self.recv_msg(sock, "WORKER_TRAIN_DONE")
            if not msg:
                raise RuntimeError(f"[CM] worker {wid} socket closed while waiting WORKER_TRAIN_DONE")

            payload = msg[1] if len(msg) > 1 else {}
            if payload.get("round") != round_idx:
                raise RuntimeError(
                    f"[CM] WORKER_TRAIN_DONE round mismatch from {wid}: got={payload.get('round')} expected={round_idx}")

            logger.info(f"[CM] Received WORKER_TRAIN_DONE from {wid} | round={round_idx}")

    def _get_local_model_shard(self, client_id):

        task = self.local_tasks.get(client_id, self.first_task)
        start = task['start_layer']
        end = task['end_layer']

        logger.info(f"Initializing Local Model Shard: Layers {start} - {end}")
        self.snapshot_start = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Snapshot BEFORE initialization:\n"
                    f"\t\t\t{self.snapshot_start}")

        # If CM is the tail and should_offload=False, it needs is_last=True
        cm_is_tail = (self.last_node.get(client_id) == self.node_id)
        is_last = (not self.should_offload) and cm_is_tail

        new_model = ModelFactory.get_model_shard(
            model_name=self.model_name,
            model_type=self.model_type,
            start_layer=start,
            end_layer=end,
            is_first=False,
            is_last=is_last,
            device=self.device,
            hf_token=self.hf_token,
            num_classes=self.num_classes
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
                              description = "Model_Shard_Load",
                              r=-1,
                              i=-1)

        return new_model, new_optimizer

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
                    logger.info(f"[REALLOC][CM] applied the parameters of the new "
                                f"range to {client_id}")

                carried = self.pending_optimizer_moments.pop(client_id, None)
                if carried:
                    restored = helpers.restore_optimizer_moments(
                        new_model, new_optimizer, carried)
                    logger.info(f"[REALLOC][CM] restored the moments of {restored} "
                                f"parameters for {client_id}")

        return self.models[client_id], self.optimizers[client_id]

    def start_training_as_worker(self):
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
            logger.info(f"\n--- Starting Round {r}/{self.global_rounds-1} ---")
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # --- LR Decay (once per round, before threads) ---
            if self.model_type == 'llm':
                if r > 0:
                    self.learning_rate *= 0.9
            elif self.model_type == 'vision':
                if r in (49, 74, 99, 124, 149):
                    self.learning_rate *= 0.1
            logger.info(f"[Round {r}] LR: {self.learning_rate}")

            # ---------------------------------------------------------
            # PHASE 1: TRAINING (Upstream Threads)
            # ---------------------------------------------------------
            for device_id, device_sock in self.tier1_devices_data.items():
                t_up = threading.Thread(
                    target=self._handle_tier1_upstream,
                    args=(device_id, device_sock, r),
                    name=f"T-Train-{device_id}",
                    daemon=True
                )
                self.threads_training_clients[device_id] = t_up
                t_up.start()


            for t in self.threads_training_clients.values():
                t.join()

            snapshot = helpers.get_local_resource_snapshot(self.device)
            logger.info(f"Snapshot: {snapshot}")

            logger.info(f"[Round {r}] Training finished. Updating weights...")

            self._wait_workers_train_done(round_idx=r)


            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            self._sync_after_round(round_idx=r)

            logger.info(f"[Round {r}] Waiting Orchestrator START_EVALUATION...")
            eval_client = self.wait_start_evaluation(round_idx=r)

            # Notify compute_nodes which clients to evaluate
            eval_clients_list = [eval_client] if eval_client else []
            for wid, wsock in self.local_compute_nodes.items():
                self.send_msg(wsock, ["EVAL_CLIENTS", {"round": r, "clients": eval_clients_list}])

            if eval_client:
                logger.info(f"[Round {r}] Starting Evaluation for {eval_client}...")
                for device_id, dev_sock in self.tier1_devices_data.items():
                    if device_id == eval_client:
                        t_eval = threading.Thread(
                            target=self._handle_evaluate,
                            args=(device_id, dev_sock, r),
                            name=f"T-Eval-{device_id}",
                            daemon=True
                        )
                        self.threads_evaluate_clients[device_id] = t_eval
                        t_eval.start()
                    else:
                        self.send_msg(dev_sock, ["SKIP_EVALUATION", {"round": r}])

                for t in self.threads_evaluate_clients.values():
                    t.join()
            else:
                # SKIP_EVALUATION: notify all local clients to skip
                for device_id, dev_sock in self.tier1_devices_data.items():
                    self.send_msg(dev_sock, ["SKIP_EVALUATION", {"round": r}])

            # --- RESOURCE READING BETWEEN ROUNDS ---
            self.collect_runtime_profiles(round_idx=r)
            self.wait_allocation_decision(round_idx=r)

            logger.info(f"[Round {r + 1}] Cycle Complete (Train + Eval).")

    def _train(self, device_id, client_sock, r):

        task = self.local_tasks.get(device_id)

        next_hop = task['next_hop']
        iterations_train = self.iterations_train[device_id]
        client_model, client_optimizer = self._get_client_model(device_id)
        client_model.train()


        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()
        # else:
        #     helpers.malloc_trim_if_possible()

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

            msg = self.recv_msg(client_sock, "FORWARD_DATA")

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

            snapshot_end = helpers.get_local_resource_snapshot(self.device)
            helpers.log_snapshots(node_id=self.node_id,
                                  run_id=self.run_id,
                                  snapshot_start=self.snapshot_start,
                                  snapshot_end=snapshot_end,
                                  r=r,
                                  i=i,
                                  description=f"After_forward")

            t_start_d2h = time.perf_counter()

            send_payload = {
                'req_id': req_id,
                'source': device_id,
                'data': None if next_hop == "LOCAL_FINISH" else output.detach().cpu(),
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


            self._handle_forward(send_payload, next_hop, device_id, i, r, req_id)

        # FedProx: free global params snapshot after round
        self.client_global_params.pop(device_id, None)

        logger.info(f"[{device_id}] Upstream Finished.")

    def _handle_tier1_upstream(self, device_id, client_sock, r):
        logger.info(f"[Train] Starting round {r}")

        try:
            self._train(device_id, client_sock, r)
        except Exception as exc:
            task = (self.local_tasks or {}).get(device_id) or {}
            helpers.report_fatal(
                exc, node_id=self.node_id, run_id=self.run_id, seed=self.seed,
                client_id=device_id, round_idx=r, batch_idx=-1,
                device=self.device, phase="train",
                layers="%s-%s" % (task.get("start_layer"), task.get("end_layer")))

        logger.info(f"[Train {device_id}] Round {r} finished")

        if self.should_offload is False:
            payload = {"client_id": device_id, "round": r}

            target_sock = self.client_train_sockets.get(device_id, self.orchestrator_sock)

            logger.debug(f"[LOCAL_FINISH][{device_id}] Sending ROUND_FINISH to Orchestrator | round={r}")
            self.send_msg(target_sock, ["ROUND_FINISH", payload])

    def _handle_forward(self, send_payload, next_hop, device_id, i, r, req_id):
        """
        OFFLOAD=True  : CM -> Workers -> CM(Tail) -> Orch -> ... -> CM -> T1
        OFFLOAD=False : CM -> Workers -> Tail(local loss) -> ... -> CM -> T1
        """

        # =========================================================
        # CASE 0: LOCAL_FINISH and Strong Cluster
        # CM is 'tail' from Strong Cluster
        # next_hop is "LOCAL_FINISH"
        # =========================================================
        if next_hop == "LOCAL_FINISH":
            req_id = send_payload['req_id']
            original_source = send_payload.get('original_source', device_id)

            client_model, client_optimizer = self._get_client_model(device_id)
            inputs, outputs = self._get_context(req_id)

            labels = send_payload['labels'].to(self.device)

            t_start_comp_loss = time.perf_counter()

            loss = self._calculate_loss(outputs, labels)
            loss_value = loss.item()
            loss.backward()

            # FedProx proximal gradient
            gp = self.client_global_params.get(device_id)
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

            snapshot_end = helpers.get_local_resource_snapshot(self.device)
            helpers.log_snapshots(node_id=self.node_id,
                                  run_id=self.run_id,
                                  snapshot_start=self.snapshot_start,
                                  snapshot_end=snapshot_end,
                                  r=r,
                                  i=i,
                                  description=f"After_backward")

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comp_loss",
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

            sock_t1 = self.tier1_devices_data[original_source]
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

            t_start_comm_backward = time.perf_counter()

            self.send_msg(sock_t1, ["BACKWARD_DATA", back_payload])

            t_end_comm_backward = time.perf_counter()
            t_comm_backward = t_end_comm_backward - t_start_comm_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
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
        # CASE 1: CM is 'tail' from Straggler Cluster
        # next_hop is tier 3
        # =========================================================
        # `_t3_` is the Aggregation Pool tag in the node ids (e.g. s1_t3_1)
        if next_hop == "tier_3_target" or "_t3_" in next_hop:
            # print("CASO 1: CM is tail and Straggler Cluster")

            target_sock = self.client_train_sockets[device_id]
            t_start_link = time.perf_counter()

            self.send_msg(target_sock, ["FORWARD_DATA", send_payload])

            t_end_link = time.perf_counter()
            t_link = t_end_link - t_start_link
            helpers.log_performance_metric(node_id=self.node_id,
                                           client_id=device_id,
                                           seed=self.seed,
                                           run_id=self.run_id,
                                           metric_name="T_link",
                                           duration=t_link,
                                           round_idx=r,
                                           batch_idx=i,
                                           req_id=req_id,
                                           delay=None
                                           )

            helpers.log_payload_size(node_id=self.node_id, run_id=self.run_id,
                                     client_id=device_id, round_idx=r,
                                     hop="T2->T3", payload=send_payload, batch_idx=i)


            t_start_wait = time.perf_counter()

            msg_t3 = self.recv_msg(target_sock, "BACKWARD_DATA_FROM_ORCH")

            t_end_wait = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_wait_offload",
                duration=(t_end_wait - t_start_wait),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            payload = msg_t3[1]
            req_id = payload.get('req_id', None)
            original_source = payload.get('original_source', None)
            grad_tensor = payload['grad'].to(self.device)
            client_model, client_optimizer = self._get_client_model(device_id)
            inputs, outputs = self._get_context(req_id)

            t_start_comp_backward = time.perf_counter()

            grad_tensor = grad_tensor.to(dtype=outputs.dtype)
            outputs.backward(grad_tensor)

            # FedProx proximal gradient
            gp = self.client_global_params.get(device_id)
            if self.mu > 0 and gp is not None:
                for name, param in client_model.named_parameters():
                    if param.grad is not None and name in gp:
                        param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=self.clip_max_norm)
            client_optimizer.step()

            helpers.sync_device(self.device)

            t_end_comp_backward = time.perf_counter()

            snapshot_end = helpers.get_local_resource_snapshot(self.device)
            helpers.log_snapshots(node_id=self.node_id,
                                  run_id=self.run_id,
                                  snapshot_start=self.snapshot_start,
                                  snapshot_end=snapshot_end,
                                  r=0,
                                  i=i,
                                  description=f"After_backward")
            t_comp_backward = t_end_comp_backward - t_start_comp_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comp_backward",
                duration=t_comp_backward,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            t_start_d2h = time.perf_counter()

            back_payload = {
                'req_id': req_id,
                'grad': inputs.grad.detach().cpu(),
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

            sock_t1 = self.tier1_devices_data[original_source]

            t_start_comm_backward = time.perf_counter()

            self.send_msg(sock_t1, ["BACKWARD_DATA", back_payload])

            t_end_comm_backward = time.perf_counter()
            t_comm_backward = t_end_comm_backward - t_start_comm_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
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
        # CASE 2: CM is not tail and can be Straggler or Strong Cluster
        # =========================================================
        else:
            # print("CASO 2: CM is not tail and can be Straggler or Strong Cluster")

            sock_next = self.worker_data_sockets[next_hop][device_id]

            t_start_comm = time.perf_counter()
            self.send_msg(sock_next, ["FORWARD_DATA_NEXT_HOP", send_payload])
            t_end_comm = time.perf_counter()
            t_comm = t_end_comm - t_start_comm

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=device_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_comm_forward",
                duration=t_comm,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay = self.delay
            )
            # =========================================================
            # CASE 2.1: CM is not tail and is from Straggler Cluster
            # =========================================================
            last_node_id = self.last_node[device_id]
            sock_last = self.worker_data_forward[last_node_id][device_id]
            if self.should_offload:
                # print(f"CASO 2.1 [{device_id}]: CM is not tail and is Straggler Cluster")

                msg_fwd = self.recv_msg(sock_last, "FORWARD_DATA_TO_TIER_3")
                payload_fwd = msg_fwd[1]

                t_start_link = time.perf_counter()

                self.send_msg(self.client_train_sockets[device_id], ["FORWARD_DATA", payload_fwd])

                t_end_link = time.perf_counter()
                t_link = t_end_link - t_start_link
                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_link",
                    duration=t_link,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                helpers.log_payload_size(node_id=self.node_id, run_id=self.run_id,
                                         client_id=device_id, round_idx=r,
                                         hop="T2->T3", payload=payload_fwd, batch_idx=i)

                t_start_wait = time.perf_counter()

                msg_t3 = self.recv_msg(self.client_train_sockets[device_id], "BACKWARD_DATA_FROM_ORCH")

                t_end_wait = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_wait_offload",
                    duration=(t_end_wait - t_start_wait),
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                payload = msg_t3[1]
                t_start_comm_backward = time.perf_counter()

                self.send_msg(sock_last, ["BACKWARD_DATA_FROM_CLUSTER_MANAGER", payload])

                t_end_comm_backward = time.perf_counter()
                t_comm_backward = t_end_comm_backward - t_start_comm_backward

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comm_backward",
                    duration=t_comm_backward,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                t_start_wait = time.perf_counter()

                msg_next_back = self.recv_msg(sock_next, "BACKWARD_DATA_FROM_PREVIOUS_NODE")

                t_end_wait = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_wait_chain",
                    duration=(t_end_wait - t_start_wait),
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )
                payload = msg_next_back[1]
                req_id = payload.get('req_id', None)
                original_source = payload.get('original_source', None)
                grad_tensor = payload['grad'].to(self.device)
                client_model, client_optimizer = self._get_client_model(device_id)
                inputs, outputs = self._get_context(req_id)

                t_start_comp_backward = time.perf_counter()

                grad_tensor = grad_tensor.to(dtype=outputs.dtype)
                outputs.backward(grad_tensor)

                # FedProx proximal gradient
                gp = self.client_global_params.get(device_id)
                if self.mu > 0 and gp is not None:
                    for name, param in client_model.named_parameters():
                        if param.grad is not None and name in gp:
                            param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=self.clip_max_norm)
                client_optimizer.step()

                helpers.sync_device(self.device)

                t_end_comp_backward = time.perf_counter()

                snapshot_end = helpers.get_local_resource_snapshot(self.device)
                helpers.log_snapshots(node_id=self.node_id,
                                      run_id=self.run_id,
                                      snapshot_start=self.snapshot_start,
                                      snapshot_end=snapshot_end,
                                      r=0,
                                      i=i,
                                      description=f"After_backward")
                t_comp_backward = t_end_comp_backward - t_start_comp_backward

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_backward",
                    duration=t_comp_backward,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )
                t_start_d2h = time.perf_counter()

                back_payload = {
                    'req_id': req_id,
                    'grad': inputs.grad.detach().cpu(),
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


                sock_t1 = self.tier1_devices_data[original_source]

                t_start_comm_backward = time.perf_counter()

                self.send_msg(sock_t1, ["BACKWARD_DATA", back_payload])

                t_end_comm_backward = time.perf_counter()
                t_comm_backward = t_end_comm_backward - t_start_comm_backward

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
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
            # CASE 2.2: CM is not tail and is from Strong Cluster
            # =========================================================
            else:
                # print("CASO 2.2: CM is not tail and is Strong Cluster")
                if  self.check:
                    print(sock_next)
                    self.check = False

                t_start_wait = time.perf_counter()

                msg_to_device = self.recv_msg(sock_next, "BACKWARD_DATA_FROM_PREVIOUS_NODE")

                t_end_wait = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_wait_chain",
                    duration=(t_end_wait - t_start_wait),
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )
                payload = msg_to_device[1]

                req_id = payload.get('req_id', None)
                original_source = payload.get('original_source', None)
                grad_tensor = payload['grad'].to(self.device)

                client_model, client_optimizer = self._get_client_model(device_id)
                inputs, outputs = self._get_context(req_id)

                t_start_comp_backward = time.perf_counter()

                grad_tensor = grad_tensor.to(dtype=outputs.dtype)
                outputs.backward(grad_tensor)

                # FedProx proximal gradient
                gp = self.client_global_params.get(device_id)
                if self.mu > 0 and gp is not None:
                    for name, param in client_model.named_parameters():
                        if param.grad is not None and name in gp:
                            param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=self.clip_max_norm)
                client_optimizer.step()

                helpers.sync_device(self.device)

                t_end_comp_backward = time.perf_counter()

                snapshot_end = helpers.get_local_resource_snapshot(self.device)
                helpers.log_snapshots(node_id=self.node_id,
                                      run_id=self.run_id,
                                      snapshot_start=self.snapshot_start,
                                      snapshot_end=snapshot_end,
                                      r=r,
                                      i=i,
                                      description=f"After_backward")
                t_comp_backward = t_end_comp_backward - t_start_comp_backward

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_backward",
                    duration=t_comp_backward,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                t_start_d2h = time.perf_counter()

                back_payload = {
                    'req_id': req_id,
                    'grad': inputs.grad.detach().cpu(),
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

                sock_t1 = self.tier1_devices_data[device_id]

                t_start_comm_backward = time.perf_counter()

                self.send_msg(sock_t1, ["BACKWARD_DATA", back_payload])

                t_end_comm_backward = time.perf_counter()
                t_comm_backward = t_end_comm_backward - t_start_comm_backward

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=device_id,
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

    def wait_start_evaluation(self, round_idx: int):
        """
        Barrier: waits for START_EVALUATION or SKIP_EVALUATION from the Orchestrator.
        Returns the eval_client id if evaluation should proceed, None if skipped.
        """
        logger.info(f"[CM] Waiting START_EVALUATION/SKIP_EVALUATION (round={round_idx})...")
        msg = self.recv_msg(self.orchestrator_data_sock)

        if not msg:
            raise RuntimeError("[CM] orchestrator_data_sock closed while waiting evaluation signal")

        command = msg[0]
        payload = msg[1] if isinstance(msg[1], dict) else {}

        if command == "SKIP_EVALUATION":
            logger.info(f"[CM] Received SKIP_EVALUATION (round={round_idx}). Skipping eval phase.")
            return None

        if command == "START_EVALUATION":
            if "round" in payload and payload["round"] != round_idx:
                logger.warning(f"[CM] START_EVALUATION round mismatch: got={payload['round']} expected={round_idx}")
            eval_client = payload.get("eval_client")
            logger.info(f"[CM] Received START_EVALUATION (round={round_idx}, eval_client={eval_client}).")
            return eval_client

        logger.warning(f"[CM] Unexpected eval signal: {command}. Treating as skip.")
        return False

    def _handle_evaluate(self, device_id, client_sock, r):
        """
        Wrapper method that manages the evaluation lifecycle for a client.
        """
        logger.info(f"[Eval {device_id}] Sending START_EVALUATION signal...")
        self.send_msg(client_sock, ["START_EVALUATION", {"round": r}])

        task = self.local_tasks.get(device_id)
        if task is None:
            next_hop = "tier_3_target"
        else:
            next_hop = task.get("next_hop", "tier_3_target")

        client_model, _ = self._get_client_model(device_id)
        client_model.eval()

        sock_to_tier3 = self.client_train_sockets.get(device_id)
        if sock_to_tier3 is None:
            raise RuntimeError(f"[Eval {device_id}] Missing dedicated socket to Tier3.")

        try:
            self._evaluate_loop(device_id, client_sock, client_model, next_hop, sock_to_tier3, r)
        finally:
            client_model.train()  # Restore training state
            logger.info(f"[Eval {device_id}] Evaluation thread finished.")

    def _evaluate_loop(self, device_id, client_sock, client_model, next_hop, sock_to_tier3, r):

        iterations_eval = self.iterations_eval[device_id]
        for i in tqdm.tqdm(range(iterations_eval), desc=f"[{self.node_id} | {device_id} | Round: {r}]"):
            msg = self.recv_msg(client_sock, "FORWARD_EVAL")
            payload = msg[1]

            processed_payload = self._process_local_inference(device_id, client_model, payload)

            self._route_evaluation_data(device_id, next_hop, processed_payload, sock_to_tier3)

    def _process_local_inference(self, device_id, model, payload):

        logger.debug(f"[Eval {device_id}] Processing local_inference command...")

        input_gpu = payload["data"].to(self.device)
        mask_raw = payload.get("attention_mask")
        pos_raw = payload.get("position_ids")
        req_id = payload.get("req_id")

        mask_gpu = mask_raw.to(self.device) if mask_raw is not None else None
        pos_gpu = pos_raw.to(self.device) if pos_raw is not None else None
        t_start_eval = time.perf_counter()
        with torch.no_grad():
            if self.model_type == "llm":
                output = model(input_gpu, attention_mask=mask_gpu, position_ids=pos_gpu)
            else:
                output = model(input_gpu)

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


        payload["data"] = output.detach().cpu()

        return payload

    def _route_evaluation_data(self, device_id, next_hop, payload, sock_to_tier3):

        def _is_direct_t3(hop):
            # `_t3_` is the Aggregation Pool tag in the node ids (e.g. s1_t3_1)
            return hop == "tier_3_target" or ("_t3_" in str(hop).lower())

        if _is_direct_t3(next_hop) or next_hop == "LOCAL_FINISH":
            self.send_msg(sock_to_tier3, ["FORWARD_EVAL", payload])
            return

        last_node_id = self.last_node.get(device_id)
        if not last_node_id:
            logger.warning(f"[Eval {device_id}] Topology inconsistency (no last_node). Routing to T3.")
            self.send_msg(sock_to_tier3, ["FORWARD_EVAL", payload])
            return

        sock_next = self.worker_data_sockets[next_hop][device_id]
        sock_last = self.worker_data_forward[last_node_id][device_id]

        if sock_next is None or sock_last is None:
            logger.error(
                f"[Eval {device_id}] Routing Error: Missing sockets. Next({next_hop})={sock_next is not None}, Last({last_node_id})={sock_last is not None}")
            self.send_msg(sock_to_tier3, ["FORWARD_EVAL", payload])
            return

        self.send_msg(sock_next, ["FORWARD_EVAL_NEXT_HOP", payload])

        # msg_back = self.recv_msg(sock_last, "FORWARD_EVAL_TO_TIER_3")
        msg_back = self.recv_msg(sock_last)
        command = msg_back[0]
        if command == "FORWARD_EVAL_TO_TIER_3":
            self.send_msg(sock_to_tier3, ["FORWARD_EVAL", msg_back[1]])
        elif command == "EVAL_METRICS":
            self.send_msg(sock_to_tier3, ["EVAL_METRICS", msg_back[1]])


        # if msg_back:
        #     self.send_msg(sock_to_tier3, ["FORWARD_EVAL", msg_back[1]])
        # else:
        #     logger.error(f"[Eval {device_id}] Chain Error: Missing response from tail {last_node_id}")

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

    def wait_and_send_start_training(self):
        logger.info(f"Waiting for START_TRAINING on socket:\n"
                    f"{self.orchestrator_data_sock.getsockname()}\n"
                    f"{self.orchestrator_data_sock.getpeername()} ")

        msg = self.recv_msg(self.orchestrator_data_sock, "START_TRAINING")

        logger.info(f"[START] Received Start Training Request: {msg}")
        logger.info(f"[START] Sending Iterations to orchestrator.")

        # my_hw = helpers.get_local_device_name(self.device)
        self.workers_hardware[self.node_id] = self.my_hw

        response_payload = {
            "iterations_train": self.iterations_train,
            "iterations_eval": self.iterations_eval,
            "hardware_map": self.workers_hardware,
            "measured_profiles": self.site_profiles
        }

        self.send_msg(self.orchestrator_data_sock, msg=["ITERATIONS", response_payload])
        payload = msg[1]
        self._handle_start_training(payload)
        return True

    def _handle_start_training(self, config):
        """
        Prepares the Manager and notifies the Agent to start.
        """
        self.global_rounds = config['global_rounds']
        self.learning_rate = config['learning_rate']
        self.mu = config.get('mu', 0.0)
        self.fedprox_cpu = config.get('fedprox_cpu', False)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.client_global_params = {}

        logger.info(f"Training configuration loaded. Rounds: {self.global_rounds}, LR: {self.learning_rate}, "
                     f"mu: {self.mu}, fedprox_cpu: {self.fedprox_cpu}, weight_decay: {self.weight_decay}")

        agent_config = {
            'global_rounds': self.global_rounds,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'mu': self.mu,
            'fedprox_cpu': self.fedprox_cpu,
        }
        compute_nodes_config = {
            'global_rounds': self.global_rounds,
            'learning_rate': self.learning_rate,
            'iterations_train': self.iterations_train,
            'iterations_eval': self.iterations_eval,
            'mu': self.mu,
            'fedprox_cpu': self.fedprox_cpu,
            'weight_decay': self.weight_decay,
            'alpha': self.alpha
        }

        # threads = []
        for device_id, dev_sock in self.tier1_devices_control.items():
            logger.info(f"Sending Start Training to {device_id}...")
            self.send_msg(dev_sock, ["START_TRAINING", agent_config])


        for worker_id, sock in self.local_compute_nodes.items():
            logger.info(f"Sending Start Training to {worker_id}...")

            self.send_msg(sock, ["START_TRAINING", compute_nodes_config])

        self.start_training_as_worker()

        logger.info(f"Training phase is done!")

    def _handle_global_weights_update(self, r):
        logger.info("[AGGR] Waiting for UPDATE_GLOBAL_WEIGHTS ...")
        msg = self.recv_msg(self.orchestrator_sock, "UPDATE_GLOBAL_WEIGHTS")
        payload = msg[1]
        logger.info(f"[AGGR] Received UPDATE_GLOBAL_WEIGHTS : {msg[0]}")
        self._apply_global_update_payload(payload, r)

    def _log_compact_chain(
        self,
        device_id: str,
        split_tuple,
        chain_nodes: list,
        chain_counts: dict,
        tier2_ignored: bool = False,
    ):
        """
        Compact log of the final pipeline per device:
          T1:0-a@<tier1_node> -> T2:<node>:a-b -> ... -> T3:<tier3_node>:c-total
        """

        parts = []

        # --- Tier 1
        t1_end = int(split_tuple.tier1_split_point)
        parts.append(f"T1:0-{t1_end}@{device_id}")

        # --- Tier 2
        cur = t1_end
        if tier2_ignored or not chain_nodes:
            parts.append("T2:SKIP")
        else:
            for nid in chain_nodes:
                lc = int(chain_counts[nid])
                nxt = cur + lc
                parts.append(f"T2:{nid}:{cur}-{nxt}")
                cur = nxt

        # --- Tier 3
        t3_node = getattr(split_tuple, "tier3_target", "T3")
        total_layers = getattr(self, "total_layers", None)

        if total_layers is None:
            parts.append(f"T3:{t3_node}:{cur}-?")
        else:
            parts.append(f"T3:{t3_node}:{cur}-{int(total_layers)}")

        logger.info(f"[CHAIN] device={device_id} | " + " -> ".join(parts))

    # [PATCH] Request Tier1 weights (parallel)
    def _collect_tier1_weights(self, round_idx: int, req_id: str, fp16: bool):
        out = {}

        def worker(cid, sock):
            self.send_msg(sock, ["REQUEST_LOCAL_WEIGHTS", {"round": round_idx, "req_id": req_id, "fp16": fp16}])
            logger.info(f"[REQUEST UPDATE] Sent Local Weights : {cid}")
            logger.info(f"[REQUEST UPDATE] Wainting Local Weights : {cid}")

            msg = self.recv_msg(sock, "LOCAL_WEIGHTS_RESPONSE")

            if not msg:
                return
            resp = msg[1]
            out[cid] = resp["pack"]

        threads = []
        for cid, sock in self.tier1_devices_data.items():
            t = threading.Thread(target=worker, args=(cid, sock), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        return out

    # [PATCH] Request worker weights (bulk, parallel)
    def _collect_worker_weights(self, round_idx: int, req_id: str, fp16: bool, client_ids: list):
        merged = {cid: [] for cid in client_ids}

        def worker(wid, sock):
            self.send_msg(sock, ["REQUEST_LOCAL_WEIGHTS_TAGGED",
                                 {"round": round_idx, "req_id": req_id, "client_ids": client_ids, "fp16": fp16}])
            msg = self.recv_msg(sock, "LOCAL_WEIGHTS_RESPONSE_TAGGED")
            logger.info(f"[REQUEST UPDATE] Received Local Weights from: {wid}")
            msg_type, resp = msg
            if resp.get("req_id") != req_id:
                logger.warning("Stale worker weights ignored (req_id mismatch)")
                return
            if resp.get("round") != round_idx:
                logger.warning("Stale worker weights ignored (round mismatch)")
                return

            if not msg:
                return

            packs_by_client = resp.get("packs_by_client", {})
            for cid, packs in packs_by_client.items():
                merged[cid].extend(packs)

        threads = []
        for wid, sock in self.local_compute_nodes.items():
            t = threading.Thread(target=worker, args=(wid, sock), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        return merged

    # [PATCH] Manager's own weights (if it holds a replica for the client)
    def _collect_manager_weights(self, round_idx: int, req_id: str, fp16: bool):
        """
        For each client the MANAGER holds a model for (self.models[cid]),
        returns 1 pack with shard signature (tier=t2, start/end from local task).
        """
        out = {}

        for cid, model in self.models.items():
            task = self.local_tasks.get(cid)
            if not task:
                continue

            start = int(task["start_layer"])
            end = int(task["end_layer"])

            sd = {}
            for k, v in model.state_dict().items():
                t = v.detach().cpu()
                if fp16 and t.is_floating_point():
                    t = t.half()
                sd[k] = t

            out[cid] = {
                "shard": {"tier": "t2", "start": start, "end": end},
                "weights": sd
            }

        return out

    # [PATCH] Full site weight collection: T1 + T2(manager) + T2(workers)
    def _collect_site_weights(self, round_idx: int, req_id: str, fp16: bool):
        """
        Returns the site weights per client, as a list of shards, each carrying its tier,
        its layer range and its state dict.
        """
        site_clients = list(self.tier1_devices_control.keys())

        # 1) Tier 1 (NodeAgents): returns packs (with shard metadata)
        t1_packs = self._collect_tier1_weights(round_idx, req_id=req_id, fp16=fp16)

        # 2) Tier 2 manager local: packs per client (if manager holds a model for that client)
        t2m_packs = self._collect_manager_weights(round_idx, req_id=req_id, fp16=fp16)

        # 3) Tier 2 workers: packs per client (list or single pack)
        t2w_packs = self._collect_worker_weights(round_idx, req_id=req_id, fp16=fp16,
                                                 client_ids=site_clients)

        shards_by_client = {cid: [] for cid in site_clients}

        for cid in site_clients:
            if cid in t1_packs:
                shards_by_client[cid].append(t1_packs[cid])
            if cid in t2m_packs:
                shards_by_client[cid].append(t2m_packs[cid])
            if cid in t2w_packs:
                # May come as a list (recommended) or single pack
                if isinstance(t2w_packs[cid], list):
                    shards_by_client[cid].extend(t2w_packs[cid])
                else:
                    shards_by_client[cid].append(t2w_packs[cid])

        logger.info(f"[WEIGHTS][SITE] collected shard-packs for {len(shards_by_client)} clients.")
        return shards_by_client

    # [PATCH] End-of-round sync in CM: REQUEST_SITE_WEIGHTS -> RESPONSE -> UPDATE_GLOBAL_WEIGHTS
    def _sync_after_round(self, round_idx: int):
        msg = self.recv_msg(self.orchestrator_sock)

        if msg[0] == "REQUEST_SITE_WEIGHTS":
            payload = msg[1] if isinstance(msg[1], dict) else {}
            req_id = payload.get("req_id", f"{self.site_name}-r{round_idx}")
            fp16 = bool(payload.get("fp16", True))

            weights_by_client = self._collect_site_weights(round_idx, req_id=req_id, fp16=fp16)
            self.send_msg(self.orchestrator_sock, ["SITE_WEIGHTS_RESPONSE", {
                "round": round_idx,
                "req_id": req_id,
                "weights_by_client": weights_by_client
            }])

            # Now wait for the global update (normal flow)
            self._handle_global_weights_update(round_idx)
            return

    def _apply_global_update_payload(self, payload, r):
        """
        Applies the aggregated weights and distributes them: the same global shards go to
        every client of the site, each tier receiving the shard that matches its range.
        """

        shards = payload.get('shards', [])
        if not isinstance(shards, list) or not shards:
            logger.warning("[AGGR][CM] No shards in payload. Skipping update.")
            return

        worker_updates = {wid: [] for wid in getattr(self, 'local_compute_nodes', {}).keys()}

        for client_id in self.all_clients:

            # ---------------------------
            # 1) Update T1 (NodeAgent)
            # ---------------------------
            t1_end = None
            try:
                alloc_line = self.current_allocation.get(client_id)
                if alloc_line is not None and hasattr(alloc_line, 'tier1_split_point'):
                    t1_end = int(getattr(alloc_line, 'tier1_split_point'))
            except Exception:
                t1_end = None

            t1_pack = None
            for p in shards:
                sh = p.get('shard', {})
                if sh.get('tier') != 't1':
                    continue
                if int(sh.get('start', -1)) != 0:
                    continue
                if t1_end is not None and int(sh.get('end', -2)) != t1_end:
                    continue
                t1_pack = p
                break

            if client_id in self.tier1_devices_data and t1_pack is not None:
                self.send_msg(self.tier1_devices_data[client_id], ["UPDATE_LOCAL_WEIGHTS", t1_pack['weights']])
            elif client_id in self.tier1_devices_data and t1_pack is None:
                logger.warning(f"[AGGR][CM] No T1 shard match for client={client_id} (t1_end={t1_end}). Trying fallback.")
                t1_pack_fallback = next(
                    (p for p in shards if p.get('shard', {}).get('tier') == 't1'),
                    None
                )
                if t1_pack_fallback is not None:
                    logger.warning(f"[AGGR][CM] Using fallback T1 shard for client={client_id}.")
                    self.send_msg(self.tier1_devices_data[client_id], ["UPDATE_LOCAL_WEIGHTS", t1_pack_fallback['weights']])
                else:
                    logger.warning(f"[AGGR][CM] No fallback T1 shard found for client={client_id}. Skipping update.")

            # ---------------------------------
            # 2) Update T2 Manager (self)
            # ---------------------------------
            task = self.local_tasks.get(client_id)
            if task is not None:
                my_start = int(task.get('start_layer'))
                my_end = int(task.get('end_layer'))

                t2m_pack = None
                for p in shards:
                    sh = p.get('shard', {})
                    if sh.get('tier') != 't2':
                        continue
                    if int(sh.get('start', -1)) == my_start and int(sh.get('end', -1)) == my_end:
                        t2m_pack = p
                        break

                if t2m_pack is not None and client_id in self.models:
                    self.models[client_id].load_state_dict(t2m_pack['weights'], strict=False)
                    if self.model_type == "llm":
                        # Keep optimizer state (Adam moments) — only refresh LR
                        for pg in self.optimizers[client_id].param_groups:
                            pg['lr'] = self.learning_rate
                    else:
                        self.optimizers[client_id] = optim.SGD(self.models[client_id].parameters(),
                                                               lr=self.learning_rate,
                                                               momentum=0.9)
                    logger.info(
                        f"[AGGR][CM] Updated local T2 shard for client={client_id} range={my_start}-{my_end}")

            # ---------------------------------
            # 3) Prepare updates for T2 Workers
            # ---------------------------------
            if getattr(self, 'local_compute_nodes', None):
                for wid in self.local_compute_nodes.keys():
                    rng = self.worker_task_ranges.get(wid, {}).get(client_id)
                    if not rng:
                        continue
                    w_start, w_end = int(rng[0]), int(rng[1])

                    pack = None
                    for p in shards:
                        sh = p.get('shard', {})
                        if sh.get('tier') != 't2':
                            continue
                        if int(sh.get('start', -1)) == w_start and int(sh.get('end', -1)) == w_end:
                            pack = p
                            break

                    if pack is None:
                        logger.warning(
                            f"[AGGR][CM] No T2 shard match for worker={wid} client={client_id} range={w_start}-{w_end}")
                        continue

                    worker_updates[wid].append({
                        'weights': pack['weights'],
                        'client_id': client_id,
                        'shard': pack.get('shard')
                    })

        # 4) Send updates to workers (one per client) and finalize with UPDATE_DONE
        if getattr(self, 'local_compute_nodes', None):
            for wid, sock in self.local_compute_nodes.items():
                for upd in worker_updates.get(wid, []):
                    self.send_msg(sock, ["UPDATE_LOCAL_WEIGHTS_TAGGED", upd])
                # Sentinel for the worker to exit the sync loop for this round
                self.send_msg(sock, ["UPDATE_DONE", {"reason": "global_update_applied"}])

    def _save_experiment_configuration(self):
        """
        Collects and persists all metadata for the current experiment.
        Includes: hyperparameters, topology, hardware, splits, and Strong/Straggler classification.
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
