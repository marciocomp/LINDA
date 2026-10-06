# @author: Marcio Lopes
import copy
import gc
import json
import math
import os
import sys
import time
import datetime
import numpy as np
import threading
import csv
import transformers
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import socket
from collections import defaultdict

from torch.fx.experimental.unification.unification_tools import first
from transformers import AutoConfig
import tqdm
sys.path.append('../../')
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.json_helper import load_json
from utils.hyperparameters import *
from utils.standardTuples.resource_vector_tuple import *
from utils.standardTuples.split_matrix_line_tuple import SplitMatrixLineTuple
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers
from utils.standardTuples.data_payload import ForwardDataTuple, BackwardGradTuple
from utils.standardTuples.resource_vector_tuple import ResourceVectorTuple


class GlobalOrchestrator(CommunicationModule):
    """
    Tier 3 Component: The Central Brain (Site 1 / Lead Cluster).
    """

    def __init__(self, seed,
                 node_id,
                 ip_address="localhost",
                 port=10000,
                 data_port=10001,
                 straggler=True,
                 hyperparameters=None,
                 topology=None,
                 # hardware_specs=None,
                 network_links=None,
                 model_name=None,
                 model_type=None,
                 alpha=1.0,
                 device="cpu",
                 global_rounds=10
                 ):

        super().__init__(node_id, ip_address)

        self.new_split_matrix = None
        self.input_embeddings_cust_gib = 0.0
        self.transf_block_standard_cust_gib = 0.0
        self.transf_block_checkpointed_cust_gib = 0.0
        self.lm_head_cost_gib = 0.0
        self.logits_tensor_cost_gib = 0.0
        self.global_rounds = global_rounds
        self.iterations_train = {}
        self.iterations_eval = {}
        self.first_iteration = {}
        self.site_end_layer = {}
        self.hyperparameters = hyperparameters
        self.hf_token = HF_TOKEN
        self.site_partial_weights = {}
        self.seed = seed
        self.alpha = alpha
        self.batch_size = hyperparameters["batch_size"]
        self.host = ip_address
        self.port = port
        self.data_port = data_port
        self.gpus_workers = {}
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.num_classes = self.hyperparameters["num_classes"]
        self.delay = 0.0
        self.snapshot_start = {}
        self.snapshot_end = {}

        self.topology = topology
        self.model_name = model_name
        self.model_type = model_type
        # self.hardware_specs = hardware_specs
        self.network_links = network_links
        self.site_should_offload = {}

        self.profiles = {}
        self.measured_profiles = {}
        self.candidate_allocation = None
        self.drift_streak = 0
        self.drift_reason = ""
        self.pending_decision = None
        self.pending_weight_packs = {}
        self.pending_optimizer_moments = {}
        self.reallocation_enabled = False
        self.realloc_min_streak = 1
        self.THROUGHPUT_TIE_TOLERANCE = 0.05
        self.global_allocation = {}
        self.network_graph = {}
        self.connected_clusters = {}
        self.connected_clusters_data = {}
        self.client_train_sockets = {}
        self.bw_lan = BW_LAN
        self.bw_wan = BW_WAN
        self.now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.enable_net_metrics(self.now)
        self.base_models_per_site = {}
        self.models = {}
        self.optimizers = {}
        self.start_layer = None
        self.end_layer = None
        self.total_layers = self._get_model_layer_count()
        self.clip_max_norm = None
        self.seq_len = 512

        if self.model_type == "llm":
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            self.clip_max_norm = 1.0
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.clip_max_norm = 1.0

        self.mu = float(self.hyperparameters.get("mu", 0.0))
        self.fedprox_cpu = bool(self.hyperparameters.get("fedprox_cpu", False))
        self.weight_decay = float(self.hyperparameters.get("weight_decay", 0.01))
        self._gen_eval_cached_data = None  # (tokenizer, list[(prompt, ref)])
        self._gen_eval_cached_cfg = None
        self._gen_eval_models = {}  # full model instance for generation eval (LLM only)
        self.VISION_BUDGET_FRACTION = 0.03

        if device == "cpu":
            self.device = torch.device(device)  # device #"cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if torch.cuda.is_available():
                gpu_id = int(device.split(":")[1])
                torch.cuda.set_device(gpu_id)
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")

        self.train_lock = threading.Lock()  # Guards the optimizer step
        logger.info(f"Orchestrator ready on {self.device} (Network: {self.host}:{self.port})")
        logger.info(f"Device Name: {helpers.get_local_device_name(self.device)}")
        logger.info(f"Target Model: {self.model_name} ({self.model_type}) | Detected Layers: {self.total_layers}")
        self.initial_snapshot = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Initial System Snapshot : {self.initial_snapshot}")


    def _accept_data_connections(self, expected_total):
        """
        Waits for dedicated connections from EACH client.
        """
        logger.info(f"[Data Plane] Waiting for {expected_total} Dedicated Client Streams...")

        # try:
        self.sock_server_data.bind((self.ip, self.data_port))
        self.sock_server_data.listen(expected_total)

        while len(self.client_train_sockets) + len(self.connected_clusters_data) < expected_total:
            client_sock, (ip, port) = self.sock_server_data.accept()
            msg = self.recv_msg(client_sock)  # Expects ["HELLO_DATA_CLIENT", client_id]

            if msg[0] == "HELLO_CLIENT_TRAIN":
                client_id = msg[1]
                self.client_train_sockets[client_id] = client_sock
                logger.info(f"[{msg[0]}] Registered Dedicated Stream: {client_id} ({ip}:{port})")

            elif msg[0] == "HELLO_DATA":
                site_id = msg[1]
                site_name = helpers.find_site_of_node(self.topology, site_id)
                self.connected_clusters_data[site_name] = client_sock
                logger.info(f"[{msg[0]}] Registered Dedicated Stream: {site_name} ({ip}:{port})")

        logger.info("[Data Plane] All Clients Connected via Dedicated Sockets!")

        # except Exception as e:
        #     logger.error(f"[Data Plane] Error accepting connections: {e}")

    def do_cluster_connections(self, expected_clusters=1):
        """
        Step 1 (Setup): Opens the socket and waits for Tier 2 Cluster Managers to connect.
        """
        logger.info(f"Waiting for {expected_clusters} Cluster Managers to connect on {self.ip}:{self.port}...")

        try:
            self.sock_server.bind((self.ip, self.port))
            self.sock_server.listen(5)

            while len(self.connected_clusters) < expected_clusters:
                client_sock, (ip, port) = self.sock_server.accept()
                logger.info(f"Incoming connection from {ip}:{port}")

                msg = self.recv_msg(client_sock, "HELLO_TIER2")

                site_name = msg[1]
                self.connected_clusters[site_name] = client_sock
                logger.info(f"Handshake OK! Registered Cluster Manager: {site_name}")

                self.perform_time_sync_handshake(client_sock, role="server")

            logger.info("All expected Cluster Managers are connected!")

        except Exception as e:
            logger.error(f"Critical Error in Orchestrator Server: {e}")
            raise e

    def _get_model_layer_count(self):
        """Discovers model metadata without loading weights."""

        if self.model_type == 'vision':
            if 'resnet50' in self.model_name:
                return 6 # (Stem + Layer1 + Layer2 + Layer3 + Layer4 + Head)
        elif self.model_type == 'llm':
            config = AutoConfig.from_pretrained(self.model_name, token=self.hf_token)
            if hasattr(config, 'num_hidden_layers'):
                return config.num_hidden_layers
            elif hasattr(config, 'n_layer'):
                return config.n_layer
        return None

    def _calculate_layer_capacity(self, ram_gb, num_clients_served=1):
        """
        Calculates capacity in BLOCKS based on the Standard (Intermediate) Cost.
        Does not account for Head/Logits here (that is handled by Tail-Awareness logic).
        """

        available_ram_gb = ram_gb * 0.90 # 10% Overhead margin

        if available_ram_gb <= 0: return 0

        if self.model_type == 'vision':
            COST_PER_BLOCK_GB = 0.25 * num_clients_served
        else:
            # Gemma-2B Standard Block (Intermediate Node)
            # 2.04 GiB per stream
            UNIT_BLOCK_COST_GIB = helpers.get_unit_block_cost_gib(
                self.model_name, self.batch_size, self.seq_len
            )
            # UNIT_BLOCK_COST_GIB = 2.2
            self.transf_block_standard_cust_gib = UNIT_BLOCK_COST_GIB
            # print(f"UNIT_BLOCK_COST_GIB: {UNIT_BLOCK_COST_GIB}")
            # UNIT_BLOCK_COST_GIB = 2.04
            COST_PER_BLOCK_GB = UNIT_BLOCK_COST_GIB * num_clients_served

        if COST_PER_BLOCK_GB == 0: return 0

        return int(available_ram_gb / COST_PER_BLOCK_GB)

    def build_network_graph(self):
        """
        Constructs the Network Graph N = (V, E) from the explicit link definition.
        Parsing 'network_links.json' to populate self.network_graph.
        """
        logger.info("Building Network Graph N from explicit link definitions...")

        # Stores edge attributes (BW and Latency) crucial for cost equations (Eq 9 and 12)
        for link in self.network_links:
            u = link['u']
            v = link['v']
            attrs = {
                'bandwidth': link['bandwidth_mbps'],
                'latency': link['latency_ms'],
                'type': link['type']
            }

            # Undirected graph (symmetric bidirectional) for communication
            self.network_graph[(u, v)] = attrs
            self.network_graph[(v, u)] = attrs

        logger.info(f"Graph N built with {len(self.network_links)} explicit connections.")

        return self.network_graph

    def generate_initial_allocation(self):
        """
        Generates the sequential allocation matrix and verifies Tier 3 feasibility. The
        pool is served in decreasing profiled throughput, memory stays the hard constraint,
        and the physical chain is never reordered.
        Returns (is_feasible, allocation_matrix).
        """

        logger.info("Generating Split Matrix (throughput-aware) & Verifying Real-World Constraints...")

        allocation_matrix = {}
        leader_id = self.node_id

        # --- COST CONSTANTS (Based on Gemma-2B Memory Table) ---
        # Values in GiB per client stream

        if self.model_type == 'llm':
            self.input_embeddings_cust_gib = helpers.get_embeddings_cost_gib(self.model_name,
                                                                             self.batch_size,
                                                                             self.seq_len)
            BLOCK_CHKPT_COST_GIB = helpers.get_block_checkpoint_cost_gib(self.model_name,
                                                                         self.batch_size,
                                                                         self.seq_len)

            HEAD_COST_GIB = helpers.get_head_cost_gib(self.model_name)
            LOGITS_COST_GIB = helpers.get_logits_cost_gib(self.model_name,
                                                          self.batch_size,
                                                          self.seq_len)
        else:
            # For Vision (ResNet, etc) — simplified logic since vision models
            UNIT_BLOCK_COST_GIB = 0.25
            BLOCK_CHKPT_COST_GIB = 0.25
            HEAD_COST_GIB = 0.1
            LOGITS_COST_GIB = 0.1

        self.transf_block_checkpointed_cust_gib = BLOCK_CHKPT_COST_GIB
        self.lm_head_cost_gib = HEAD_COST_GIB
        self.logits_tensor_cost_gib = LOGITS_COST_GIB
        # Accumulated load for the Orchestrator
        total_tier3_load_gib = 0.0

        # Fixed Tier 1 layers
        t1_layers_fixed = 1

        for site_name, site_data in self.topology['sites'].items():
            tier1_devices = site_data.get('tier1', {})
            tier2_servers = site_data.get('tier2', {})

            num_active_clients = len(tier1_devices)
            if num_active_clients == 0:
                continue

            layers_remaining = self.total_layers - t1_layers_fixed
            t2_distribution = {}

            # --- MEMORY BUDGET OF EACH NODE, IN BLOCKS ---
            chain_order = [s_conf['id'] for s_conf in tier2_servers.values()]
            capacity = {}
            for s_id in chain_order:
                profile = self.profiles.get(s_id)
                capacity[s_id] = self._calculate_layer_capacity(
                    profile.ram if profile else 0.0, num_active_clients)

            # --- THROUGHPUT PRIORITY ---

            offered = {s_id: 0 for s_id in chain_order}
            budget = layers_remaining
            for s_id in chain_order:
                if budget <= 0:
                    logger.warning(
                        f"Site {site_name}: {s_id} gets no block, because the layers "
                        f"ran out before the chain did ({len(chain_order)} nodes for "
                        f"{layers_remaining} layers).")
                    continue
                if capacity[s_id] <= 0:
                    logger.warning(f"Site {site_name}: {s_id} cannot hold a single "
                                   f"block and stays out of the path.")
                    continue
                offered[s_id] = 1
                budget -= 1

            def _throughput(node_id):
                profile = self.profiles.get(node_id)
                return profile.flops if profile else 0.0


            speed_class, cls, reference = {}, 0, None
            for node_id in sorted(chain_order, key=lambda n: -_throughput(n)):
                speed = _throughput(node_id)
                if reference is None:
                    reference = speed
                elif reference - speed > self.THROUGHPUT_TIE_TOLERANCE * reference:
                    cls += 1
                    reference = speed
                speed_class[node_id] = cls


            def _tail_capacity(node_id):
                if self.model_type != 'llm':
                    return capacity[node_id]
                profile = self.profiles.get(node_id)
                room = (profile.ram if profile else 0.0) * 0.90 \
                    - num_active_clients * (HEAD_COST_GIB + LOGITS_COST_GIB)
                if room <= 0:
                    return 0
                return min(capacity[node_id],
                           int(room / (num_active_clients * BLOCK_CHKPT_COST_GIB)))

            # Service order: who is offered blocks first. Ties keep topology order.
            service_order = sorted(chain_order,
                                   key=lambda n: (speed_class[n], chain_order.index(n)))

            # Offer by throughput priority, bounded by what each node's memory holds.
            for s_id in service_order:
                if budget <= 0:
                    break
                room = max(0, capacity[s_id] - offered[s_id])
                take = min(budget, room)
                offered[s_id] += take
                budget -= take


            if budget == 0:
                trial, spare_budget, capped = dict(offered), 0, set()
                for _ in range(len(chain_order)):
                    tail = None
                    for s_id in chain_order:
                        if trial[s_id] > 0:
                            tail = s_id
                    if tail is None or tail in capped:
                        break

                    allowed = _tail_capacity(tail)
                    if trial[tail] <= allowed:
                        break
                    if allowed < 1:

                        logger.info(f"Node {tail} cannot terminate the chain at all; "
                                    f"Tier 3 will, and the plain offer stands.")
                        spare_budget = -1
                        break

                    logger.info(f"Node {tail} would end the chain with {trial[tail]} "
                                f"blocks but can only terminate {allowed}; offering "
                                f"the rest to the other nodes of the site.")
                    spare_budget += trial[tail] - allowed
                    trial[tail] = allowed
                    capped.add(tail)

                    for s_id in service_order:
                        if spare_budget <= 0:
                            break
                        if s_id in capped:
                            continue
                        room = max(0, capacity[s_id] - trial[s_id])
                        take = min(spare_budget, room)
                        trial[s_id] += take
                        spare_budget -= take

                if spare_budget == 0:
                    offered = trial
                else:
                    logger.info(f"Site {site_name}: the pool cannot absorb what the "
                                f"tail would give up, so Tier 3 terminates the path "
                                f"and the plain offer stands.")

            # Logged in CHAIN order, so the line reads as the chain really is.
            logger.info(
                "Site %s: chain %s | offered by throughput priority (served %s)"
                % (site_name,
                   [(n, "%.1f TFLOPS" % _throughput(n), "class %d" % speed_class[n],
                     "cap %d" % capacity[n], "offered %d" % offered[n])
                    for n in chain_order],
                   " > ".join(service_order)))

            # --- WATERFALL ALLOCATION IN TIER 2 ---

            for s_id in chain_order:
                if layers_remaining <= 0: break

                profile = self.profiles.get(s_id)
                ram_total = profile.ram if profile else 0.0

                # Safety margin (OS + Fragmentation)
                ram_available = ram_total * 0.90

                capacity_std = capacity[s_id]
                to_take = min(layers_remaining, capacity_std, offered[s_id])

                # --- HEAD AWARENESS LOGIC (Precise) ---
                # Check if this node is a candidate to be the TAIL (take all remaining layers)
                if self.model_type == 'llm' and to_take == layers_remaining and to_take > 0:


                    needed_tail_gib = num_active_clients * (
                            (to_take * BLOCK_CHKPT_COST_GIB) +
                            HEAD_COST_GIB +
                            LOGITS_COST_GIB
                    )

                    if ram_available < needed_tail_gib:
                        logger.warning(
                            f"  -> Node {s_id} (Avail: {ram_available:.2f} GiB) "
                            f"fits blocks but triggers OOM on Head/Logits "
                            f"(Need: {needed_tail_gib:.2f} GiB). Forcing offload to Tier 3."
                        )

                        to_take = (max(1, layers_remaining - 1) if layers_remaining > 1
                                   else 0)
                    else:
                        logger.info(f"Node {s_id} accepted as Local Tail (Load: {needed_tail_gib:.2f} GiB).")

                if to_take > 0:
                    t2_distribution[s_id] = to_take
                    layers_remaining -= to_take
                    logger.info(f"Node {s_id} (RAM: {ram_total:.1f}GB | "
                                f"{_throughput(s_id):.1f} TFLOPS): "
                                f"Took {to_take} layers. (Cap: {capacity_std}, "
                                f"offered: {offered[s_id]})")
                else:
                    logger.warning(
                        f"Node {s_id} (RAM: {ram_total:.1f}GB): Skipped (Capacity Full, 0, or Reserved for Head).")

            # --- SPILLBACK: nothing leaves the site while the site has room ---

            for s_id in chain_order:
                if layers_remaining <= 0:
                    break
                spare = capacity[s_id] - t2_distribution.get(s_id, 0)
                if spare <= 0:
                    continue

                to_take = min(layers_remaining, spare)

                position = chain_order.index(s_id)
                ends_the_chain = all(t2_distribution.get(later, 0) == 0
                                     for later in chain_order[position + 1:])
                if (self.model_type == 'llm' and ends_the_chain
                        and to_take == layers_remaining):
                    to_take = min(to_take,
                                  max(0, _tail_capacity(s_id) - t2_distribution.get(s_id, 0)))

                if to_take > 0:
                    t2_distribution[s_id] = t2_distribution.get(s_id, 0) + to_take
                    layers_remaining -= to_take
                    logger.info(f"Node {s_id}: took {to_take} more layer(s) so that "
                                f"they stay inside the site (now {t2_distribution[s_id]}).")

            # The chain stays in topology order after the spillback.
            t2_distribution = {s_id: t2_distribution[s_id]
                               for s_id in chain_order if s_id in t2_distribution}
            for device_key, device_conf in tier1_devices.items():
                allocation_matrix[device_conf['id']] = SplitMatrixLineTuple(
                    tier1_split_point=t1_layers_fixed,
                    tier2_distribution=t2_distribution,
                    tier3_target=leader_id
                )

                if layers_remaining > 0:
                    logger.info(
                        f"Device {device_conf['id']}: {layers_remaining} layers + HEAD overflowed to Tier 3. [Candidate: STRAGGLER]")
                else:
                    logger.info(f"Device {device_conf['id']}: All layers absorbed by Tier 2. [Candidate: STRONG]")

            # --- ORCHESTRATOR LOAD CALCULATION ---
            if layers_remaining > 0:


                logger.info(f"Site {site_name}: Offloading {layers_remaining} layers + Head to Tier 3.")

                site_load_gib = num_active_clients * (
                        (layers_remaining * BLOCK_CHKPT_COST_GIB) +
                        HEAD_COST_GIB +
                        LOGITS_COST_GIB
                )
                total_tier3_load_gib += site_load_gib
            else:
                logger.info(f"Site {site_name}: Local Finish (Zero load on Tier 3).")

        # --- FINAL TIER 1 FEASIBILITY CHECK ---
        leader_profile = self.profiles.get(self.node_id)

        t3_capacity_gib = leader_profile.ram
        is_feasible = total_tier3_load_gib < t3_capacity_gib
        self.global_allocation = allocation_matrix

        logger.info("=== Final Allocation Matrix ===")
        for dev, line in self.global_allocation.items():
            logger.info(f"{dev}: T1={line.tier1_split_point} | T2={line.tier2_distribution} | T3={line.tier3_target}")

        logger.info("=== Tier 3 Viability Check ===")
        logger.info(f"  Total Requested Load: {total_tier3_load_gib:.2f} GiB")
        logger.info(f"  Orchestrator Capacity: {t3_capacity_gib:.2f} GiB")
        logger.info(f"  Allocation Feasible: {is_feasible}")

        if not is_feasible:
            logger.error(
                f"CRITICAL: Tier 3 OOM Predicted! Req: {total_tier3_load_gib:.2f} vs Avail: {t3_capacity_gib:.2f} GiB")
            return False, allocation_matrix

        return True, allocation_matrix

    def generate_initial_allocation_old(self):
        """
        Memory-driven allocation, kept verbatim as the published behaviour, superseded as
        the default by `generate_initial_allocation`.
        Returns (is_feasible, allocation_matrix).
        """

        logger.info("Generating Split Matrix & Verifying Real-World Constraints...")

        allocation_matrix = {}
        leader_id = self.node_id
        # new
        # --- COST CONSTANTS (Based on Gemma-2B Memory Table) ---
        # Values in GiB per client stream

        # BLOCK_CHKPT_COST_GIB = 1.0 #1.27  # Optimized (Tail / Tier 3 nodes)
        # HEAD_COST_GIB = 5.0 #5.87  # Static Weights + Optimizer
        # LOGITS_COST_GIB = 6.0 #8.95  # Dynamic Logits + Buffers (Softmax/Loss)

        if self.model_type == 'llm':
            self.input_embeddings_cust_gib = helpers.get_embeddings_cost_gib(self.model_name,
                                                                             self.batch_size,
                                                                             self.seq_len)
            BLOCK_CHKPT_COST_GIB = helpers.get_block_checkpoint_cost_gib(self.model_name,
                                                                         self.batch_size,
                                                                         self.seq_len)

            HEAD_COST_GIB = helpers.get_head_cost_gib(self.model_name)
            LOGITS_COST_GIB = helpers.get_logits_cost_gib(self.model_name,
                                                          self.batch_size,
                                                          self.seq_len)
        else:
            # For Vision (ResNet, etc) — simplified logic since vision models
            UNIT_BLOCK_COST_GIB = 0.25
            BLOCK_CHKPT_COST_GIB = 0.25
            HEAD_COST_GIB = 0.1
            LOGITS_COST_GIB = 0.1



        self.transf_block_checkpointed_cust_gib = BLOCK_CHKPT_COST_GIB
        self.lm_head_cost_gib = HEAD_COST_GIB
        self.logits_tensor_cost_gib = LOGITS_COST_GIB
        # Accumulated load for the Orchestrator
        total_tier3_load_gib = 0.0

        # Fixed Tier 1 layers
        # t1_layers_fixed = 3
        t1_layers_fixed = 1 #max(1, int(self.total_layers * 0.15))

        for site_name, site_data in self.topology['sites'].items():
            tier1_devices = site_data.get('tier1', {})
            tier2_servers = site_data.get('tier2', {})

            num_active_clients = len(tier1_devices)
            if num_active_clients == 0:
                continue

            layers_remaining = self.total_layers - t1_layers_fixed
            t2_distribution = {}

            # --- WATERFALL ALLOCATION IN TIER 2 ---
            for s_key, s_conf in tier2_servers.items():
                s_id = s_conf['id']
                if layers_remaining <= 0: break

                profile = self.profiles.get(s_id)
                ram_total = profile.ram if profile else 0.0

                # Safety margin (OS + Fragmentation)
                ram_available = ram_total * 0.90

                # 1. Raw capacity in Standard Blocks
                capacity_std = self._calculate_layer_capacity(ram_total, num_active_clients)
                to_take = min(layers_remaining, capacity_std)

                # --- HEAD AWARENESS LOGIC (Precise) ---
                # Check if this node is a candidate to be the TAIL (take all remaining layers)
                if self.model_type == 'llm' and to_take == layers_remaining:

                    # Real cost to be a Tail Node:
                    # (remaining blocks w/ checkpoint) + (static Head) + (dynamic Logits)
                    # multiplied by number of clients (parallel streams)

                    needed_tail_gib = num_active_clients * (
                            (to_take * BLOCK_CHKPT_COST_GIB) +
                            HEAD_COST_GIB +
                            LOGITS_COST_GIB
                    )

                    if ram_available < needed_tail_gib:
                        logger.warning(
                            f"  -> Node {s_id} (Avail: {ram_available:.2f} GiB) "
                            f"fits blocks but triggers OOM on Head/Logits "
                            f"(Need: {needed_tail_gib:.2f} GiB). Forcing offload to Tier 3."
                        )
                        # Can't be Tail; demote to intermediate node (Standard Cost already validated)
                        to_take = max(0, layers_remaining - 1)
                    else:
                        logger.info(f"Node {s_id} accepted as Local Tail (Load: {needed_tail_gib:.2f} GiB).")

                if to_take > 0:
                    t2_distribution[s_id] = to_take
                    layers_remaining -= to_take
                    logger.info(f"Node {s_id} (RAM: {ram_total:.1f}GB): "
                                f"Took {to_take} layers. (Cap: {capacity_std})")
                else:
                    logger.warning(
                        f"Node {s_id} (RAM: {ram_total:.1f}GB): Skipped (Capacity Full, 0, or Reserved for Head).")


            for device_key, device_conf in tier1_devices.items():
                allocation_matrix[device_conf['id']] = SplitMatrixLineTuple(
                    tier1_split_point=t1_layers_fixed,
                    tier2_distribution=t2_distribution,
                    tier3_target=leader_id
                )

                if layers_remaining > 0:
                    logger.info(
                        f"Device {device_conf['id']}: {layers_remaining} layers + HEAD overflowed to Tier 3. [Candidate: STRAGGLER]")
                else:
                    logger.info(f"Device {device_conf['id']}: All layers absorbed by Tier 2. [Candidate: STRONG]")

            # --- ORCHESTRATOR LOAD CALCULATION ---
            if layers_remaining > 0:

                logger.info(f"Site {site_name}: Offloading {layers_remaining} layers + Head to Tier 3.")

                site_load_gib = num_active_clients * (
                        (layers_remaining * BLOCK_CHKPT_COST_GIB) +
                        HEAD_COST_GIB +
                        LOGITS_COST_GIB
                )
                total_tier3_load_gib += site_load_gib
            else:
                logger.info(f"Site {site_name}: Local Finish (Zero load on Tier 3).")

        # --- FINAL TIER 1 FEASIBILITY CHECK ---
        leader_profile = self.profiles.get(self.node_id)

        t3_capacity_gib = leader_profile.ram
        is_feasible = total_tier3_load_gib < t3_capacity_gib
        self.global_allocation = allocation_matrix

        logger.info("=== Final Allocation Matrix ===")
        for dev, line in self.global_allocation.items():
            logger.info(f"{dev}: T1={line.tier1_split_point} | T2={line.tier2_distribution} | T3={line.tier3_target}")

        logger.info("=== Tier 3 Viability Check ===")
        logger.info(f"  Total Requested Load: {total_tier3_load_gib:.2f} GiB")
        logger.info(f"  Orchestrator Capacity: {t3_capacity_gib:.2f} GiB")
        logger.info(f"  Allocation Feasible: {is_feasible}")

        if not is_feasible:
            logger.error(
                f"CRITICAL: Tier 3 OOM Predicted! Req: {total_tier3_load_gib:.2f} vs Avail: {t3_capacity_gib:.2f} GiB")
            return False, allocation_matrix

        return True, allocation_matrix

    def _should_offload(self, site_name: str, site_allocation: dict) -> bool:
        """
        True when, for any client of the site, the layers allocated in Tier 2 fall short of
        what remains after the Tier 1 split, so layers are left for Tier 3.
        """
        if not site_allocation:
            logger.warning(f"_should_offload: site '{site_name}' has no allocation. Forcing offload for safety.")
            return True

        for client_id, split_tuple in site_allocation.items():
            try:
                t1 = int(split_tuple.tier1_split_point)
                required_after_t1 = max(0, self.total_layers - t1)
                t2_taken = sum(
                    int(v) for v in split_tuple.tier2_distribution.values()) if split_tuple.tier2_distribution else 0

                if t2_taken < required_after_t1:
                    logger.info(
                        f"_should_offload[{site_name}]: client {client_id} needs offload "
                        f"(T2={t2_taken} < needed={required_after_t1})."
                    )
                    return True
            except Exception as e:
                logger.exception(f"_should_offload[{site_name}]: error evaluating client {client_id}: {e}")
                return True

        logger.info(f"_should_offload[{site_name}]: STRONG (no offload).")
        return False

    def _is_straggler(self, site_id):
        """
        True when the site's chain (Tier 1 + Tier 2) cannot hold every hidden layer, so the
        Orchestrator has to run the remainder.
        """
        site_data = self.topology['sites'].get(site_id)

        tier1_devices = site_data.get('tier1', {})

        rep_device_id = next(iter(tier1_devices.values()))['id']

        alloc_line = self.global_allocation[rep_device_id]

        layers_processed = alloc_line.tier1_split_point

        if alloc_line.tier2_distribution:
            layers_processed += sum(alloc_line.tier2_distribution.values())

        self.site_end_layer[site_id] = layers_processed
        return layers_processed < self.total_layers

    def _get_local_model_shard(self, site_id):
        """
        Loads the model shard.
        Assumes self.start_layer and self.end_layer have already been set externally.
        # """
        site_split_point = self.site_end_layer[site_id]
        logger.info(
            f"Initializing Tier 3 Shard | Layers: "
            f"{site_split_point} -> {self.end_layer if self.end_layer else 'End'}")

        self.snapshot_start = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Snapshot BEFORE initialization:\n"
                    f"\t\t\t{self.snapshot_start}")

        logger.info(f"Loading Complementary Model for {site_id}: Layers {site_split_point} -> End")

        new_model = ModelFactory.get_model_shard(
            model_name=self.model_name,
            model_type=self.model_type,
            start_layer=site_split_point,
            end_layer= self.end_layer,
            is_first=False,
            is_last=True,
            device=self.device,
            hf_token=HF_TOKEN,
            num_classes=self.num_classes
        )
        if self.model_type == "llm":
            new_optimizer = optim.AdamW(new_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        else:
            new_optimizer = optim.SGD(new_model.parameters(), lr=self.learning_rate, momentum=0.9)

        logger.info("Model Shard initialized successfully.")
        self.snapshot_end = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Snapshot AFTER initialization:\n"
                    f"\t\t\t{self.snapshot_end}")

        helpers.log_snapshots(node_id=self.node_id,
                              run_id=self.now,
                              snapshot_start=self.snapshot_start,
                              snapshot_end=self.snapshot_end,
                              description="Model_Shard_Load",
                              r=-1,
                              i=-1)

        return new_model, new_optimizer

    def _get_client_model(self, client_id, site_id):
        """
        Returns (model, optimizer) specific to the client.
        Creates a copy from the site template if needed.
        """
        with self.train_lock:

            if client_id not in self.models:
                logger.info(f"Creating Orchestrator Replica for Client: {client_id} (Site: {site_id})")

                new_model, new_optimizer = self._get_local_model_shard(site_id)
                new_model.to(self.device)

                self.models[client_id] = new_model
                self.optimizers[client_id] = new_optimizer

                # A segment rebuilt after a re-allocation starts from the
                # pretrained checkpoint, so the parameters cut for its new range
                # are applied here rather than lost.
                held = self.pending_weight_packs.pop(client_id, None)
                if held:
                    new_model.load_state_dict(held, strict=False)
                    logger.info(f"[REALLOC] applied the parameters of the new range "
                                f"to {client_id}")

                # And the moments of the layers Tier 3 kept, so only the layers the
                # site reclaimed restart their AdamW estimates.
                carried = self.pending_optimizer_moments.pop(client_id, None)
                if carried:
                    restored = helpers.restore_optimizer_moments(
                        new_model, new_optimizer, carried)
                    logger.info(f"[REALLOC] restored the moments of {restored} "
                                f"parameters for {client_id}")

        return self.models[client_id], self.optimizers[client_id]

    def distribute_initial_context(self):
        """
        Distributes the initial allocation and the global metadata (client map, seed,
        batch size, model and offload mode) to every site.
        """
        logger.info("Step 2b: Computing Client IDs and Distributing Context...")

        global_client_map = {}
        counter = 0

        sorted_sites = sorted(self.topology['sites'].keys())
        for site_key in sorted_sites:
            site_data = self.topology['sites'][site_key]
            tier1_devices = site_data.get('tier1', {})
            for dev_key in sorted(tier1_devices.keys()):
                real_id = tier1_devices[dev_key]['id']
                global_client_map[real_id] = counter
                counter += 1

        total_clients = counter
        # total_clients = 4

        logger.info(f"Global Client Mapping Generated (Total: {total_clients})")

        site_payloads = {}
        for node_id, split_tuple in self.global_allocation.items():
            site_name = helpers.find_site_of_node(self.topology, node_id)
            if site_name:
                if site_name not in site_payloads:
                    site_payloads[site_name] = {'allocation': {}, 'client_map': {}}

                site_payloads[site_name]['allocation'][node_id] = split_tuple

                if node_id in global_client_map:
                    site_payloads[site_name]['client_map'][node_id] = global_client_map[node_id]

        logger.info(f"Connected_clusters: {self.connected_clusters}")

        for site_name, data in site_payloads.items():
            if site_name not in self.connected_clusters:
                logger.warning(f"   -> Site {site_name} has data but Manager is NOT connected!")
                return False

            # Decide mode (offload vs local-finish) BEFORE sending context
            should_offload = self._should_offload(site_name, data['allocation'])
            self.site_should_offload[site_name] = should_offload
            self._is_straggler(site_name)

            final_msg = {
                'allocation': data['allocation'],
                'client_map': data['client_map'],
                'total_clients': total_clients,
                'seed': self.seed,
                'alpha': self.alpha,
                'batch_size': self.batch_size,
                'model_name': self.model_name,
                'model_type': self.model_type,
                'should_offload': should_offload,
                'run_id': self.now,
                "learning_rate": self.learning_rate,
                'num_classes': self.num_classes

            }

            logger.info(f"Payload {site_name}: \n\tFinal Message: {final_msg}\n")
            logger.info(f"Sending context to {site_name} (Clients: {list(data['client_map'].values())})")

            target_socket = self.connected_clusters[site_name]
            msg = ["INITIAL_ALLOCATION", final_msg]
            self.send_msg(target_socket, msg)

            logger.info("Waiting Cluster Response...")
            ack = self.recv_msg(target_socket, "INITIAL_ALLOCATION_COMPLETE")
            logger.info(f"Received Cluster ({site_name}) confirmation: {ack}")

        return True

    def _match_specs(self, hostname):
        """
        Matches a hostname (e.g., 'server_a40_2') to specs key (e.g., 'server_a40').
        Uses the loaded hardware_specs dictionary.
        """
        # for key, specs in self.hardware_specs.items():
        #     if key == hostname:
        #         return specs
        return None


    def _shrink_vision_budget(self, ram_gib):
        """The memory the allocator is allowed to see, scaled down for a vision run."""
        if str(self.model_type) == "llm":
            return ram_gib
        return float(ram_gib) * float(self.VISION_BUDGET_FRACTION)

    def _profile_node(self, node_config):
        """
        Implementation of ProfileNode(v). Returns a ResourceVector built from what the node
        measured on its own hardware; a node with no reading gets a zeroed vector, never a
        declared one.
        """
        node_id = node_config.get("id")

        measured = self.measured_profiles.get(node_id)
        if not measured:
            logger.warning(f"[HW] {node_id} did not report a measurement; "
                           f"it enters the allocation with no capacity.")
            return ResourceVectorTuple(flops=0.0, ram=0.0)

        return ResourceVectorTuple(flops=float(measured.get("flops", 0.0)),
                                   ram=float(measured.get("ram", 0.0)))

    def request_site_profiles(self):
        """
        Algorithm 1, Resource Discovery: each site measures its own devices and its Cluster
        Manager carries the readings up. Runs before the allocation; Tier 1 is absent,
        since its agents connect later.
        """
        logger.info("Resource Discovery: requesting measured profiles from the sites...")

        for site_name, sock in self.connected_clusters.items():
            self.send_msg(sock, ["REQUEST_SITE_PROFILE", {}])

        for site_name, sock in self.connected_clusters.items():
            msg = self.recv_msg(sock, "SITE_PROFILE_REPORT")
            if not msg:
                logger.error(f"[HW] no profile report from {site_name}.")
                continue
            reported = (msg[1] or {}).get("profiles", {}) or {}
            self.measured_profiles.update(reported)
            logger.info(f"[HW] {site_name} reported {len(reported)} measured profiles")

        self.measured_profiles[self.node_id] = helpers.measure_local_resources(
            self.node_id, self.device)

        if str(self.model_type) != "llm":
            for node_id, measured in self.measured_profiles.items():
                read = float(measured.get("ram", 0.0))
                measured["ram"] = self._shrink_vision_budget(read)
                logger.info(f"[HW] vision run: {node_id} budget {read:.2f} -> "
                            f"{measured['ram']:.2f} GiB "
                            f"({self.VISION_BUDGET_FRACTION:.0%} of the reading)")

        return self.measured_profiles

    def run_profiling(self):
        """
        Executes profiling for ALL nodes in the topology.
        Populates self.profiles with ResourceVectors (RAM, FLOPS).
        """
        logger.info(f"Executing Algorithm 1: Profiling Network Nodes...")

        sites = self.topology.get("sites", {})


        for site_name, site_data in sites.items():
            # --- Profile Tier 3 Leader (Aggregation Pool) ---
            tier3 = site_data.get("tier3",{})
            for server_key, server_conf in tier3.items():
                node_id = server_conf["id"]
                r_vector = self._profile_node(server_conf)
                if r_vector:
                    self.profiles[node_id] = r_vector
                    logger.info(f"[T3] {node_id}: {r_vector.ram:.2f} GiB measured | "
                                f"{r_vector.flops:.2f} TFLOPS measured")

            tier2 = site_data.get("tier2", {})
            for server_key, server_conf in tier2.items():
                node_id = server_conf["id"]

                r_vector = self._profile_node(server_conf)

                if r_vector:
                    self.profiles[node_id] = r_vector
                    logger.info(f"[T2] {node_id}: {r_vector.ram:.2f} GiB measured | "
                                f"{r_vector.flops:.2f} TFLOPS measured")
                else:
                    logger.warning(f"[T2] {node_id}: Profiling Failed (Unknown Hardware). Defaulting to Weak.")
                    self.profiles[node_id] = ResourceVectorTuple(flops=1.0, ram=2.0)


        logger.info(f"Profiling Complete. {len(self.profiles)} nodes scanned.")
        return self.profiles

    def request_runtime_profiles(self, round_idx):
        """
        Reads the infrastructure again between rounds, asking each site rather than its
        Compute Nodes. Called after the evaluation, where nothing is in flight. Only reads:
        deciding happens elsewhere.
        """
        started = time.perf_counter()
        readings = {}

        sent_synced = self.get_synced_time()
        for site_name, sock in self.connected_clusters.items():
            self.send_msg(sock, ["REQUEST_SITE_PROFILE", {"round": round_idx}])

        breakdown = {}
        for site_name, sock in self.connected_clusters.items():
            msg = self.recv_msg(sock, "SITE_PROFILE_REPORT")
            if not msg:
                logger.error(f"[HW] round {round_idx}: no report from {site_name}.")
                continue
            payload = msg[1] or {}
            site_readings = payload.get("profiles", {}) or {}
            readings.update(site_readings)

            arrived = payload.get("t_arrived_synced")
            rendezvous = max(0.0, float(arrived) - sent_synced) if arrived else None
            collect = float(payload.get("t_collect_s", 0.0))
            breakdown[site_name] = {
                "rendezvous_s": rendezvous,
                "collect_s": collect,
                "read_at_s": time.perf_counter() - started,
                "readings": len(site_readings),
            }

        own_started = time.perf_counter()
        own = helpers.measure_local_resources(self.node_id, self.device, runtime=True)
        own["round"] = round_idx
        readings[self.node_id] = own
        own_s = time.perf_counter() - own_started

        elapsed = time.perf_counter() - started
        self._log_runtime_profiles(round_idx, readings, elapsed)
        self._log_discovery_breakdown(round_idx, breakdown, own_s, elapsed)
        self.measured_profiles.update(readings)
        return readings

    def _log_discovery_breakdown(self, round_idx, breakdown, own_s, elapsed):
        """
        Says where the discovery time went, per site. Most of the wall-clock total is the
        rendezvous with the slowest site, not the readings, which take milliseconds.
        """
        waits = [b["rendezvous_s"] for b in breakdown.values()
                 if b["rendezvous_s"] is not None]
        slowest = max(waits) if waits else 0.0
        work = sum(b["collect_s"] for b in breakdown.values()) + own_s
        logger.info(f"[HW] round {round_idx}: discovery {elapsed:.2f}s = "
                    f"{slowest:.2f}s waiting for the last site to arrive + "
                    f"{work:.2f}s of measurement (own reading {own_s:.3f}s)")

        try:
            path = helpers.results_path(self.now, "reallocation")
            os.makedirs(path, exist_ok=True)
            out = os.path.join(path, f"discovery_seed_{self.seed}_{self.now}.csv")
            new_file = not os.path.exists(out)
            with open(out, "a", newline="") as fh:
                writer = csv.writer(fh)
                if new_file:

                    writer.writerow(["round", "site", "n_readings",
                                     "t_rendezvous_s", "t_site_collect_s",
                                     "t_report_read_s", "t_discovery_total_s"])
                for site_name, b in sorted(breakdown.items()):
                    rendezvous = b["rendezvous_s"]
                    writer.writerow([
                        round_idx, site_name, b["readings"],
                        "" if rendezvous is None else round(rendezvous, 4),
                        round(b["collect_s"], 4), round(b["read_at_s"], 4),
                        round(elapsed, 4)])
                writer.writerow([round_idx, self.node_id, 1, 0.0,
                                 round(own_s, 4), round(elapsed, 4),
                                 round(elapsed, 4)])
        except Exception as e:
            logger.warning(f"[HW] could not record the discovery breakdown of "
                           f"round {round_idx}: {e}")

    def _log_runtime_profiles(self, round_idx, readings, elapsed):
        """
        Records what each node reported, compared with the vector the allocation in force
        was computed from.
        """
        moved = []
        for node_id, reading in sorted(readings.items()):
            before = self.profiles.get(node_id)
            if before is None or reading.get("ram") is None:
                continue
            d_ram = float(reading["ram"]) - before.ram
            d_flops = float(reading.get("flops", 0.0)) - before.flops
            if abs(d_ram) >= 0.25 or (before.flops > 0
                                      and abs(d_flops) / before.flops >= 0.05):
                moved.append(f"{node_id} {before.ram:.2f}->{reading['ram']:.2f} GiB, "
                             f"{before.flops:.1f}->{reading.get('flops', 0.0):.1f} TFLOPS")

        logger.info(f"[HW] round {round_idx}: {len(readings)} readings in {elapsed:.2f}s"
                    + (f" | moved: {'; '.join(moved)}" if moved else " | nothing moved"))

        try:
            path = helpers.results_path(self.now, "reallocation")
            os.makedirs(path, exist_ok=True)
            out = os.path.join(path, f"profiles_seed_{self.seed}_{self.now}.csv")
            new_file = not os.path.exists(out)
            with open(out, "a", newline="") as fh:
                writer = csv.writer(fh)
                if new_file:
                    writer.writerow(["round", "node_id", "device", "device_name",
                                     "ram_allocatable_gib", "flops_tflops",
                                     "free_now_gib", "held_by_this_process_gib",
                                     "held_by_others_gib", "total_gib",
                                     "ram_at_allocation_gib", "flops_at_allocation",
                                     "t_measure_s", "t_probe_s", "t_round_trip_s"])
                for node_id, reading in sorted(readings.items()):
                    before = self.profiles.get(node_id)
                    writer.writerow([
                        round_idx, node_id, reading.get("device"),
                        reading.get("device_name"),
                        round(float(reading.get("ram", 0.0)), 4),
                        round(float(reading.get("flops", 0.0)), 4),
                        round(float(reading.get("free_now_gib", 0.0)), 4),
                        round(float(reading.get("held_by_this_process_gib", 0.0)), 4),
                        round(float(reading.get("reserved_gib", 0.0)), 4),
                        round(float(reading.get("total_gib", 0.0)), 4),
                        round(before.ram, 4) if before else "",
                        round(before.flops, 4) if before else "",
                        round(float(reading.get("t_measure_s", 0.0)), 4),
                        round(float(reading.get("t_probe_s", 0.0)), 4),
                        round(elapsed, 4)])
        except Exception as e:
            logger.warning(f"[HW] could not record the readings of round {round_idx}: {e}")

    @staticmethod
    def _allocation_signature(matrix):
        """
        Comparable form of a split matrix. The Tier 3 remainder is whatever the Tier 1 split
        and the Tier 2 counts leave over, so comparing those two compares the whole
        placement.
        """
        return tuple(sorted(
            (client_id,
             int(line.tier1_split_point),
             tuple(sorted((node, int(blocks))
                          for node, blocks in (line.tier2_distribution or {}).items())))
            for client_id, line in (matrix or {}).items()))

    def update_profiles_from_readings(self, readings):
        """
        Brings the resource state up to date with what the nodes just reported. A node that
        did not report keeps the vector it had.
        """

        block = self.transf_block_standard_cust_gib or 0.0
        ram_floor = max(block, 0.25)

        moved = []
        for node_id, reading in (readings or {}).items():
            if node_id not in self.profiles or reading.get("ram") is None:
                continue
            before = self.profiles[node_id]
            after = ResourceVectorTuple(
                flops=float(reading.get("flops", before.flops)),
                ram=self._shrink_vision_budget(float(reading["ram"])))

            worth_reporting = abs(after.ram - before.ram) >= ram_floor
            if not worth_reporting and before.flops > 0:
                worth_reporting = (abs(after.flops - before.flops) / before.flops
                                   >= self.THROUGHPUT_TIE_TOLERANCE)
            if worth_reporting:
                moved.append(node_id)

            self.profiles[node_id] = after
        return moved

    def significant_drift(self):
        """
        Algorithm 2, Phase 1: recomputes the allocation from the current profiles and
        compares it with the one in force. An infeasible candidate is refused, and the
        allocation in force stays in force.
        """
        self.candidate_allocation = None
        self.drift_reason = ""

        if not self.reallocation_enabled:
            self.drift_reason = "disabled"
            return False

        in_force = dict(self.global_allocation)
        signature_in_force = self._allocation_signature(in_force)

        feasible, candidate = self.generate_initial_allocation()
        self.global_allocation = in_force          # nothing is distributed yet

        if not feasible:
            self.drift_streak = 0
            self.drift_reason = "candidate_infeasible"
            logger.error("[DRIFT] the recomputed allocation does not fit Tier 3; "
                         "keeping the one in force.")
            return False

        if self._allocation_signature(candidate) == signature_in_force:
            self.drift_streak = 0
            self.drift_reason = "no_change"
            return False

        self.drift_streak += 1
        if self.drift_streak < self.realloc_min_streak:
            self.drift_reason = "holding_%d_of_%d" % (self.drift_streak,
                                                      self.realloc_min_streak)
            logger.info(f"[DRIFT] a different allocation was computed "
                        f"({self.drift_streak}/{self.realloc_min_streak} rounds); holding.")
            return False

        self.candidate_allocation = candidate
        self.drift_reason = "reallocate"
        return True

    @staticmethod
    def _to_global_key_vision(key, start):
        """
        Absolute name of a vision shard key. A vision shard is an nn.Sequential whose
        blocks are numbered from zero, so the shard's first layer is the offset.
        """
        index, _, tail = str(key).partition(".")
        return f"{start + int(index)}.{tail}" if index.isdigit() else key

    @staticmethod
    def _cut_range_vision(global_state, start, end):
        """
        The parameters a vision segment covering [start, end) would hold. Inverse of
        `_to_global_key_vision`: blocks are renumbered from the segment's first layer.
        """
        out = {}
        for block in range(int(start), int(end)):
            prefix = "%d." % block
            for global_key, value in global_state.items():
                if global_key.startswith(prefix):
                    out["%d.%s" % (block - int(start), global_key[len(prefix):])] = value
        return out

    def _global_key(self, key, start):
        """Absolute parameter name, by the naming the model in use follows."""
        if str(self.model_type) != "llm":
            return self._to_global_key_vision(key, start)
        return self._to_global_key(key, start)

    def _cut_for_range(self, global_state, start, end, is_first=False, is_last=False):
        """The parameters of [start, end), by the naming the model in use follows."""
        if str(self.model_type) != "llm":
            return self._cut_range_vision(global_state, start, end)
        return self._cut_range(global_state, start, end, is_first=is_first, is_last=is_last)

    def _global_state_from_packs(self):
        """
        The averaged parameters under absolute names, rebuilt from this round's packs. This
        is what lets a re-allocation cut the parameters of a new range from what the round
        already produced, without moving weights over the network.
        """
        state = {}
        for pack in (self.global_averaged_weights or {}).get("shards", []):
            start = int(pack.get("shard", {}).get("start", 0))
            for key, value in (pack.get("weights") or {}).items():
                state.setdefault(self._global_key(key, start), value)
        return state

    @staticmethod
    def _cut_range(global_state, start, end, is_first=False, is_last=False):
        """
        The parameters a segment covering [start, end) would hold. Inverse of
        `_to_global_key`: blocks are renumbered from the segment's first layer, the
        embedding travels with the first segment and the head with the last.
        """
        out = {}
        for layer in range(int(start), int(end)):
            prefix = "layers.%d." % layer
            for global_key, value in global_state.items():
                if global_key.startswith(prefix):
                    out["internal_model.layers.%d.%s"
                        % (layer - int(start), global_key[len(prefix):])] = value
        for global_key, value in global_state.items():
            if is_first and global_key.startswith("embed_tokens."):
                out["internal_model." + global_key] = value
            if is_last and global_key.startswith("norm."):
                out["internal_model." + global_key] = value
            if is_last and global_key.startswith("lm_head."):
                out[global_key] = value
        return out

    def _site_plan(self, matrix, site_name):
        """What a site looks like under `matrix`: its chain, its ranges, its mode."""
        line = None
        for client_id, candidate in (matrix or {}).items():
            if helpers.find_site_of_node(self.topology, client_id) == site_name:
                line = candidate
                break
        if line is None:
            return None

        t1 = int(line.tier1_split_point)
        chain = [(node, int(blocks))
                 for node, blocks in (line.tier2_distribution or {}).items()
                 if int(blocks) > 0]
        held = sum(blocks for _, blocks in chain)
        residual = self.total_layers - t1 - held

        ranges, cursor = [], t1
        for node, blocks in chain:
            ranges.append((node, cursor, cursor + blocks))
            cursor += blocks

        return {"site": site_name, "t1_split_point": t1, "chain": chain,
                "ranges": ranges, "tier3_start": cursor,
                "should_offload": residual > 0, "tier3_layers": residual,
                "clients": [c for c in (matrix or {})
                            if helpers.find_site_of_node(self.topology, c) == site_name]}

    @staticmethod
    def _node_ranges(plan):
        """
        (start, end, terminates) for each node of a site's plan. A node keeps its block
        count and still holds different layers when a node before it gains or loses one.
        """
        if not plan:
            return {}
        actives = [node for node, _, _ in plan["ranges"]]
        return {node: (int(start), int(end),
                       bool(node == actives[-1] and not plan["should_offload"]))
                for node, start, end in plan["ranges"]}

    def _chain_membership(self, matrix, site_name):
        plan = self._site_plan(matrix, site_name)
        return tuple(node for node, _ in plan["chain"]) if plan else ()

    def apply_new_allocation(self, round_idx):
        """
        Puts the candidate placement in force, weights included, sent to each site and
        handed down its chain. A plan that would change which nodes form a chain is refused.
        """
        candidate = self.candidate_allocation
        if not candidate:
            self._tell_sites_to_keep(round_idx)
            return False

        for site_name in self.connected_clusters:
            if self._chain_membership(candidate, site_name) != \
                    self._chain_membership(self.global_allocation, site_name):
                logger.error(f"[REALLOC] round {round_idx}: the plan changes which "
                             f"nodes form the chain of {site_name}; refused, and the "
                             f"allocation in force is kept.")
                self.candidate_allocation = None
                self.drift_reason = "membership_change_refused"
                self._tell_sites_to_keep(round_idx)
                return False

        started = time.perf_counter()
        global_state = self._global_state_from_packs()
        if not global_state:
            logger.error("[REALLOC] no averaged parameters in hand; refused.")
            self.candidate_allocation = None
            self.drift_reason = "no_weights_to_cut"
            self._tell_sites_to_keep(round_idx)
            return False

        updated = []
        for site_name, sock in self.connected_clusters.items():
            plan = self._site_plan(candidate, site_name)
            if plan is None:
                continue

            after = self._node_ranges(plan)
            before = self._node_ranges(self._site_plan(self.global_allocation, site_name))
            changed = [node for node in after if after[node] != before.get(node)]

            if not changed:
                self.send_msg(sock, ["KEEP_ALLOCATION", {"round": round_idx}])
                logger.info(f"[REALLOC] round {round_idx}: {site_name} is unchanged; "
                            f"it keeps the placement it has.")
                continue

            packs = {node: self._cut_for_range(global_state, after[node][0], after[node][1],
                                               is_last=after[node][2])
                     for node in changed}

            self.send_msg(sock, ["UPDATE_ALLOCATION", {
                "round": round_idx,
                "site": site_name,
                "t1_split_point": plan["t1_split_point"],
                "chain": plan["chain"],
                "ranges": plan["ranges"],
                "should_offload": plan["should_offload"],
                "packs": packs,
            }])
            updated.append(site_name)
            logger.info(f"[REALLOC] round {round_idx}: sent {site_name} its new plan "
                        f"{plan['chain']} offload={plan['should_offload']}; "
                        f"parameters for {changed}")

        applied_everywhere = True
        for site_name in updated:
            sock = self.connected_clusters[site_name]
            ack = self.recv_msg(sock, "UPDATE_ALLOCATION_DONE")
            if not ack or not (ack[1] or {}).get("applied"):
                reason = (ack[1] or {}).get("reason", "no answer") if ack else "no answer"
                logger.error(f"[REALLOC] {site_name} did not apply the new plan "
                             f"({reason}).")
                applied_everywhere = False

        if not applied_everywhere:

            logger.error("[REALLOC] the new plan was not applied everywhere; Tier 3 "
                         "keeps the allocation in force.")
            self.candidate_allocation = None
            self.drift_reason = "partially_applied"
            return False

        self._adopt_own_side(candidate, global_state)

        self.global_allocation = candidate
        self.candidate_allocation = None
        elapsed = time.perf_counter() - started

        helpers.log_performance_metric(
            node_id=self.node_id, client_id="ALL", seed=self.seed, run_id=self.now,
            metric_name="T_reallocation_apply", duration=elapsed, round_idx=round_idx,
            batch_idx=-1, req_id=None, delay=None)
        logger.info(f"[REALLOC] round {round_idx}: new allocation in force ({elapsed:.2f}s)")
        return True

    def _tell_sites_to_keep(self, round_idx):
        """
        Says that nothing changes, which still has to be said: every site waits for exactly
        one decision per round boundary.
        """
        for site_name, sock in self.connected_clusters.items():
            self.send_msg(sock, ["KEEP_ALLOCATION", {"round": round_idx}])

    def _adopt_own_side(self, candidate, global_state):
        """
        Updates what Tier 3 itself holds under the new placement: its replicas are dropped
        and the parameters of the new range are set aside, to be applied when rebuilt.
        """
        for site_name in list(self.site_should_offload.keys()):
            plan = self._site_plan(candidate, site_name)
            if plan is None:
                continue


            unchanged = (self.site_should_offload.get(site_name) == plan["should_offload"]
                         and self.site_end_layer.get(site_name) == plan["tier3_start"])

            was_offloading = self.site_should_offload.get(site_name)
            old_start = self.site_end_layer.get(site_name)
            self.site_should_offload[site_name] = plan["should_offload"]
            self.site_end_layer[site_name] = plan["tier3_start"]
            if unchanged:
                continue

            for client_id in plan["clients"]:

                model = self.models.pop(client_id, None)
                optimizer = self.optimizers.pop(client_id, None)
                if (model is not None and optimizer is not None and was_offloading
                        and plan["should_offload"] and old_start is not None):
                    held = helpers.capture_optimizer_moments(model, optimizer)
                    kept, left = helpers.carry_optimizer_moments(
                        held, old_start, plan["tier3_start"], self.total_layers,
                        prefix=helpers.shard_layer_prefix(self.model_type))
                    self.pending_optimizer_moments[client_id] = kept
                    logger.info(f"[REALLOC] {client_id}: Tier 3 carried the moments of "
                                f"{len(kept)} parameters over its new range "
                                f"{plan['tier3_start']}-{self.total_layers}, {left} "
                                f"left behind with the layers the site reclaimed.")
                if plan["should_offload"]:
                    self.pending_weight_packs[client_id] = self._cut_for_range(
                        global_state, plan["tier3_start"], self.total_layers,
                        is_last=True)

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def evaluate_drift(self, round_idx, readings):
        """
        Runs Phase 1 of Algorithm 2 for one round boundary and records the decision.
        Applying it is a separate step, because the weights move with the layers.
        """
        started = time.perf_counter()

        capacity_before = self._capacity_by_node()
        moved = self.update_profiles_from_readings(readings)
        capacity_after = self._capacity_by_node()
        capacity_delta = ["%s %d->%d" % (node, capacity_before.get(node, 0),
                                         capacity_after.get(node, 0))
                          for node in sorted(set(capacity_before) | set(capacity_after))
                          if capacity_before.get(node) != capacity_after.get(node)]


        if round_idx >= self.global_rounds - 1:
            self.candidate_allocation = None
            self.drift_reason = "last_round"
            drifted = False
            logger.info(f"[DRIFT] round {round_idx} is the last one; no placement is "
                        f"computed, since no round would run it.")
        else:
            drifted = self.significant_drift()

        elapsed = time.perf_counter() - started

        before = self._describe_allocation(self.global_allocation)
        after = self._describe_allocation(self.candidate_allocation) if drifted else ""
        relocated = self._relocated_nodes(self.candidate_allocation) if drifted else []

        if drifted:
            logger.info(f"[DRIFT] round {round_idx}: the resource state calls for a "
                        f"different placement"
                        + (f" ({'; '.join(capacity_delta)} blocks)"
                           if capacity_delta else "") + ".")
            logger.info(f"[DRIFT]   in force : {before}")
            logger.info(f"[DRIFT]   candidate: {after}")
            logger.info(f"[DRIFT]   relocated: {', '.join(relocated) or 'none'}")
        else:
            logger.info(f"[DRIFT] round {round_idx}: no change worth re-planning "
                        f"({len(moved)} node(s) reported a different figure"
                        + (f"; capacity {'; '.join(capacity_delta)}"
                           if capacity_delta else "; capacity unchanged") + ").")


        self.pending_decision = {
            "round": round_idx, "drifted": drifted, "moved": moved,
            "relocated": relocated, "capacity_delta": capacity_delta,
            "before": before, "after": after, "t_decision_s": elapsed,
        }
        return drifted

    def _capacity_by_node(self):
        """
        How many blocks each Tier 2 node's budget affords right now, by the same function
        the allocator uses and sized for the site's client count.
        """
        out = {}
        for site_data in (self.topology.get('sites') or {}).values():
            clients = len(site_data.get('tier1', {}) or {})
            if clients == 0:
                continue
            for conf in (site_data.get('tier2', {}) or {}).values():
                node_id = conf['id']
                profile = self.profiles.get(node_id)
                out[node_id] = self._calculate_layer_capacity(
                    profile.ram if profile else 0.0, clients)
        return out

    def _relocated_nodes(self, candidate):
        """
        Nodes whose layers would actually move, in chain order. Not the same set as the
        nodes whose measurement moved; this is the set `apply_new_allocation` sends
        parameters to.
        """
        if not candidate:
            return []
        out = []
        for site_name in (self.topology.get('sites') or {}):
            after = self._node_ranges(self._site_plan(candidate, site_name))
            before = self._node_ranges(self._site_plan(self.global_allocation, site_name))
            out.extend(node for node in sorted(set(after) | set(before))
                       if after.get(node) != before.get(node))
        return out

    def _describe_allocation(self, matrix):
        """One readable line per site: who holds how many blocks, and what is left."""
        if not matrix:
            return ""
        seen, parts = set(), []
        for client_id, line in sorted(matrix.items()):
            site = helpers.find_site_of_node(self.topology, client_id)
            if site in seen:
                continue
            seen.add(site)
            held = sum(int(v) for v in (line.tier2_distribution or {}).values())
            left = self.total_layers - int(line.tier1_split_point) - held
            chain = " ".join("%s=%d" % (node, int(blocks))
                             for node, blocks in (line.tier2_distribution or {}).items())
            parts.append("%s T1=%d %s T3=%d"
                         % (site, int(line.tier1_split_point), chain, left))
        return " | ".join(parts)

    def _flush_drift_decision(self, applied, t_apply_s):
        """
        One row per round boundary, saying what was seen, decided and done. Called after the
        apply, so `reason` is the final one. Does nothing if no decision is pending.
        """
        record = self.pending_decision
        self.pending_decision = None
        if not record:
            return

        round_idx = record["round"]
        try:
            path = helpers.results_path(self.now, "reallocation")
            os.makedirs(path, exist_ok=True)
            out = os.path.join(path, f"decisions_seed_{self.seed}_{self.now}.csv")
            new_file = not os.path.exists(out)
            with open(out, "a", newline="") as fh:
                writer = csv.writer(fh)
                if new_file:
                    writer.writerow(["round", "reallocation_enabled", "drifted",
                                     "applied", "reason", "nodes_reading_moved",
                                     "nodes_relocated", "capacity_delta_blocks",
                                     "allocation_in_force", "allocation_candidate",
                                     "t_decision_s", "t_apply_s"])
                writer.writerow([round_idx, self.reallocation_enabled,
                                 record["drifted"], applied, self.drift_reason,
                                 ";".join(sorted(record["moved"])),
                                 ";".join(record["relocated"]),
                                 ";".join(record["capacity_delta"]),
                                 record["before"], record["after"],
                                 round(record["t_decision_s"], 4),
                                 round(t_apply_s, 4)])
        except Exception as e:
            logger.warning(f"[DRIFT] could not record the decision of round {round_idx}: {e}")

    def start_global_training_loop(self):
        """
        Controls the global lifecycle: Rounds -> Training -> Aggregation -> Evaluation.
        """
        logger.info("=== Starting Global Orchestrator Data Plane ===")

        # Accept data-plane connections (previously in CKA profiling)
        expected_clusters = len(self.connected_clusters)
        expected_clients = 0
        if self.topology and 'sites' in self.topology:
            for site_name, site_data in self.topology['sites'].items():
                if site_name in self.connected_clusters:
                    tier1_devices = site_data.get('tier1', {})
                    expected_clients += len(tier1_devices)
        self._accept_data_connections(expected_total=expected_clusters + expected_clients)

        training_config = {
            "global_rounds": self.global_rounds,
            "learning_rate": self.hyperparameters["learning_rate"],
            "weight_decay": self.weight_decay,
            "mu": self.mu,
            "fedprox_cpu": self.fedprox_cpu,
            "momentum": 0.9
        }

        # Broadcast initial configuration to clusters
        for site_name, sock in self.connected_clusters.items():
            if site_name != "s1_t3_1":  # Avoids loopback if present
                target_sock = self.connected_clusters_data.get(site_name)
                if target_sock:
                    self.send_msg(target_sock, ["START_TRAINING", training_config])

                    msg = self.recv_msg(target_sock, "ITERATIONS")
                    if msg:
                        payload = msg[1]

                        if isinstance(payload, dict) and "hardware_map" in payload:
                            self.iterations_train.update(payload["iterations_train"])
                            self.iterations_eval.update(payload["iterations_eval"])
                            self.gpus_workers.update(payload["hardware_map"])

                            for node_id, reading in (payload.get("measured_profiles")
                                                     or {}).items():
                                if node_id in self.profiles or not reading:
                                    continue
                                self.measured_profiles[node_id] = reading
                                self.profiles[node_id] = ResourceVectorTuple(
                                    flops=float(reading.get("flops", 0.0)),
                                    ram=float(reading.get("ram", 0.0)))
                                logger.info(f"[T1] {node_id}: "
                                            f"{self.profiles[node_id].ram:.2f} GiB measured | "
                                            f"{self.profiles[node_id].flops:.2f} TFLOPS measured")
                            logger.info(
                                f"[{site_name}] Hardware report received: {len(payload['hardware_map'])} nodes.")
                        else:
                            self.iterations_train.update(payload)

        self.gpus_workers[self.node_id] = helpers.get_local_device_name(self.device)
        self._save_experiment_configuration()


        # ==============================================================================
        # GLOBAL ROUND LOOP
        # ==============================================================================
        for r in range(self.global_rounds):
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            t_start_global_cycle = time.perf_counter()

            round_start_time = time.perf_counter()
            logger.info(f"\n{'=' * 20} Starting Global Round {r}/{self.global_rounds-1} {'=' * 20}")

            # --- LR Decay (once per round, before threads) ---
            if self.model_type == 'llm':
                if r > 0:
                    self.learning_rate *= 0.9
            elif self.model_type == 'vision':
                if r in (49, 74, 99, 124, 149):
                    self.learning_rate *= 0.1
            logger.info(f"[Round {r}] LR: {self.learning_rate}")

            threads = []
            for client_id, sock in self.client_train_sockets.items():
                site_name = helpers.find_site_of_node(self.topology, client_id)
                client_iters = self.iterations_train.get(client_id, 0)

                if self.site_should_offload[site_name]:
                    self.models[client_id], self.optimizers[client_id] = self._get_client_model(client_id, site_name)

                t = threading.Thread(
                    target=self._guarded_cluster_stream,
                    args=(site_name, sock, client_iters, client_id, r),
                    name=f"Train-{client_id}-R{r}"
                )
                t.daemon = True
                t.start()
                threads.append(t)

            for t in threads: t.join()

            train_duration = time.perf_counter() - round_start_time

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id="ALL",
                seed=self.seed,
                run_id=self.now,
                metric_name="T_train_phase",
                duration=train_duration,
                round_idx=r,
                batch_idx=-1,
                req_id=None,
                delay=None
            )

            snapshot = helpers.get_local_resource_snapshot(self.device)
            logger.info(f"Snapshot: {snapshot}")

            logger.info(f"[Round {r}] Training finished ({train_duration:.2f}s). Aggregating...")


            t_start_collect_w_time = time.perf_counter()
            self._collect_site_weights(round_idx=r, fp16=True)
            t_end_collect_w_time = time.perf_counter()
            t_collect = t_end_collect_w_time - t_start_collect_w_time
            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id="None",
                seed=self.seed,
                run_id=self.now,
                metric_name="T_collect_weight",
                duration=t_collect,
                round_idx=r,
                batch_idx=-1,
                req_id="None",
                delay=None
            )

            t_start_aggregation = time.perf_counter()
            self._perform_global_aggregation(r)
            t_end_aggregation = time.perf_counter()
            t_aggregation = t_end_aggregation - t_start_aggregation

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id="None",
                seed=self.seed,
                run_id=self.now,
                metric_name="T_aggregation",
                duration=t_aggregation,
                round_idx=r,
                batch_idx=-1,
                req_id="None",
                delay=None
            )


            t_start_distribute_w = time.perf_counter()
            self._distribute_aggregated_weights()
            t_end_distribute_w = time.perf_counter()
            t_distribute_w = t_end_distribute_w - t_start_distribute_w
            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id="None",
                seed=self.seed,
                run_id=self.now,
                metric_name="T_distribution_weight",
                duration=t_distribute_w,
                round_idx=r,
                batch_idx=-1,
                req_id="None",
                delay=None
            )

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            t_start_evaluate = time.perf_counter()
            self._perform_global_evaluation(current_round=r, training_time=train_duration)
            t_end_evaluate = time.perf_counter()
            t_evaluate = t_end_evaluate - t_start_evaluate

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id="None",
                seed=self.seed,
                run_id=self.now,
                metric_name="T_global_evaluate",
                duration=t_evaluate,
                round_idx=r,
                batch_idx=-1,
                req_id="None",
                delay=None
            )


            t_start_profiling = time.perf_counter()
            readings = self.request_runtime_profiles(round_idx=r)
            t_discovery = time.perf_counter() - t_start_profiling

            t_start_decision = time.perf_counter()
            self.evaluate_drift(round_idx=r, readings=readings)
            t_decision = time.perf_counter() - t_start_decision


            t_start_apply = time.perf_counter()
            applied = self.apply_new_allocation(round_idx=r)
            t_apply = time.perf_counter() - t_start_apply

            self._flush_drift_decision(applied=applied, t_apply_s=t_apply)

            for metric_name, duration in (
                    ("T_resource_discovery", t_discovery),
                    ("T_realloc_decision", t_decision),
                    ("T_realloc_between_rounds",
                     time.perf_counter() - t_start_profiling)):
                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id="None",
                    seed=self.seed,
                    run_id=self.now,
                    metric_name=metric_name,
                    duration=duration,
                    round_idx=r,
                    batch_idx=-1,
                    req_id="None",
                    delay=None
                )

            t_end_global_cycle = time.perf_counter()
            t_global_cycle = t_end_global_cycle - t_start_global_cycle

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id="None",
                seed=self.seed,
                run_id=self.now,
                metric_name="T_global_cycle",
                duration=t_global_cycle,
                round_idx=r,
                batch_idx=-1,
                req_id="None",
                delay=None
            )

            logger.info(f"=== Round {r} Completed ===\n")

    def _guarded_cluster_stream(self, site_id, sock, iterations, client_id, rounds):
        """
        `_handle_cluster_stream` in a thread that reports instead of dying quietly. Tier 3
        carries the residual layers of every straggler client, so an out-of-memory here
        would hang the barrier rather than show up as a result.
        """
        try:
            self._handle_cluster_stream(site_id, sock, iterations, client_id, rounds)
        except Exception as exc:
            helpers.report_fatal(
                exc, node_id=self.node_id, run_id=self.now, seed=self.seed,
                client_id=client_id, round_idx=rounds, batch_idx=-1,
                device=self.device, phase="offload",
                layers="%s-%s" % (self.start_layer, self.end_layer))

    def _handle_cluster_stream(self, site_id, sock, iterations, client_id, rounds):
        """
        Runs training for one round for one client. With should_offload the Orchestrator
        processes the remaining layers and the head; without it, only waits for
        ROUND_FINISH.
        """
        # client_model, client_optimizer = self._get_client_model(client_id, site_id)
        # self.models[client_id] = client_model
        # self.optimizers[client_id] = client_optimizer


        logger.info(f"[{client_id}] Started Round {rounds} (Target: {iterations} iterations)")
        t_start_cluster = time.time()

        should_offload = self.site_should_offload.get(site_id, True)

        if not should_offload:
            logger.info(f"[{client_id}] LOCAL_FINISH. Waiting ROUND_FINISH...")
            self.recv_msg(sock, "ROUND_FINISH")
            logger.info(f"[{client_id}] Finished Round {rounds}. Waiting at barrier.")
            return

        self._train(site_id, sock, iterations, client_id, rounds)
        gc.collect()
        logger.info(f"[{client_id}] Finished Round {rounds}. Waiting at barrier.")

        t_end_cluster = time.time()
        cluster_latency = t_end_cluster - t_start_cluster

        helpers.log_performance_metric(
            node_id=self.node_id,
            client_id=client_id,
            seed=self.seed,
            run_id=self.now,
            metric_name="total_cluster_latency",
            duration=cluster_latency,
            round_idx=rounds,
            batch_idx=-1,
            req_id=client_id,
            delay=None
        )

    def _train(self, site_id, sock, iterations, client_id, r):
        should_offload = self.site_should_offload.get(site_id, True)
        if not should_offload:
            logger.warning(
                f"[TRAIN {client_id}] should_offload=False, but _train() was called. Aborting training in Orchestrator.")
            return

        logger.info(f"[TRAIN {client_id}] Stream Started | Mode: OFFLOAD (Layers + Head/Loss)")
        client_model, client_optimizer = self._get_client_model(client_id, site_id)
        client_model.train()


        # Apply current LR (already decayed in round loop)
        for param_group in client_optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        # FedProx: save global weights before local training
        global_params = None
        if self.mu > 0:
            if self.fedprox_cpu:
                global_params = {n: p.detach().cpu() for n, p in client_model.named_parameters()}
            else:
                global_params = {n: p.clone().detach() for n, p in client_model.named_parameters()}

        snapshot_start = helpers.get_local_resource_snapshot(self.device)
        for i in tqdm.tqdm(range(iterations), desc=f"[{self.node_id} | {client_id} | Round: {r}]"):
            msg = self.recv_msg(sock, "FORWARD_DATA")

            payload = msg[1]
            req_id = payload['req_id']

            input_raw = payload['data']
            labels_raw = payload['labels']
            mask_raw = payload.get('attention_mask', None)
            pos_raw = payload.get('position_ids', None)

            original_source = payload['original_source']

            inputs = input_raw.to(self.device)
            labels = labels_raw.to(self.device)
            masks = mask_raw.to(self.device) if mask_raw is not None else None
            pos_ids = pos_raw.to(self.device) if pos_raw is not None else None

            if not inputs.requires_grad:
                inputs.requires_grad_(True)

            client_optimizer.zero_grad()

            t_start_offload = time.perf_counter()

            if self.model_type == 'llm':
                outputs = client_model(inputs,
                                       attention_mask=masks,
                                       position_ids=pos_ids)
            else:
                outputs = client_model(inputs)

            loss = self._calculate_loss(outputs, labels)

            loss_value = loss.item()

            loss.backward()

            # FedProx proximal gradient: grad += mu * (w - w_global)
            if self.mu > 0 and global_params is not None:
                for name, param in client_model.named_parameters():
                    if param.grad is not None and name in global_params:
                        param.grad.data.add_(self.mu * (param.data - global_params[name].data.to(param.device)))

            del outputs, loss
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=self.clip_max_norm)
            client_optimizer.step()


            helpers.sync_device(self.device)

            t_end_offload = time.perf_counter()
            t_offload = t_end_offload - t_start_offload

            snapshot_end = helpers.get_local_resource_snapshot(self.device)
            helpers.log_snapshots(node_id=self.node_id,
                                  run_id=self.now,
                                  snapshot_start=snapshot_start,
                                  snapshot_end=snapshot_end,
                                  r=r,
                                  i=i,
                                  description=f"After_forward")

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.now,
                metric_name="T_offload",
                duration=t_offload,
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
                run_id=self.now,
                client_id=original_source,
                group_id=group_id,
                round_idx=r,
                iteration_idx=i,
                model_type=self.model_type,
                metrics=metrics,
                alpha=self.alpha,
                seed=self.seed
            )


            t_start_d2h = time.perf_counter()

            grad_cpu = inputs.grad.detach().cpu()

            t_end_d2h = time.perf_counter()

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.now,
                metric_name="T_d2h_backward",
                duration=(t_end_d2h - t_start_d2h),
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )

            response_payload = {
                'req_id': req_id,
                'grad': grad_cpu,
                'original_source': original_source,
            }

            helpers.log_payload_size(node_id=self.node_id, run_id=self.now,
                                     client_id=original_source, round_idx=r,
                                     hop="T3->T2", payload=grad_cpu, batch_idx=i)

            t_start_offload_backward = time.perf_counter()

            self.send_msg(sock, ["BACKWARD_DATA_FROM_ORCH", response_payload])

            t_end_offload_backward = time.perf_counter()
            t_offload_backward = t_end_offload_backward - t_start_offload_backward

            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=original_source,
                seed=self.seed,
                run_id=self.now,
                metric_name="T_offload_backward",
                duration=t_offload_backward,
                round_idx=r,
                batch_idx=i,
                req_id=req_id,
                delay=None
            )
            snapshot_end = helpers.get_local_resource_snapshot(self.device)
            helpers.log_snapshots(node_id=self.node_id,
                                  run_id=self.now,
                                  snapshot_start=snapshot_start,
                                  snapshot_end=snapshot_end,
                                  r=r,
                                  i=i,
                                  description=f"After_backward")


            logger.debug(f"Gradients was sent to [{site_id}]")

        gc.collect()

    @staticmethod
    def _to_global_key(key, start):
        if key.startswith("internal_model.layers."):
            idx, _, tail = key[len("internal_model.layers."):].partition(".")
            if idx.isdigit():
                return f"layers.{start + int(idx)}.{tail}"
        if key.startswith("internal_model."):
            return key[len("internal_model."):]
        return key

    def _perform_global_aggregation(self, r):
        """
        Layer-wise weighted FedAvg. Packs are translated to absolute parameter names and
        averaged one parameter at a time, so every client holding a parameter contributes
        even when the sites cut the model at different layers.
        """
        if str(self.model_type) != "llm":
            return self._perform_vision_aggregation(r)

        t_start_agg = time.perf_counter()
        logger.info(f"=== Starting Global Aggregation (Layer-wise Weighted FedAvg) | Round {r} ===")

        if not hasattr(self, "site_partial_weights") or self.site_partial_weights is None:
            self.site_partial_weights = {}

        all_clients = list(self.site_partial_weights.keys())
        acc = {}                    # global key -> float32 accumulator
        wsum = defaultdict(float)   # global key -> summed sample weight
        first_non_tensor = {}       # global key -> first value seen (ints/bools/buffers)
        templates = {}              # shard signature -> [(shard key, global key), ...]

        for client_id in all_clients:
            n_iters = self.iterations_train.get(client_id, 0)
            n_samples = n_iters * self.batch_size
            if n_samples <= 0:
                logger.warning(f"Client {client_id} has 0 samples. Ignored.")
                continue

            partial_packs = self.site_partial_weights.get(client_id, [])
            if partial_packs is None:
                partial_packs = []
            elif isinstance(partial_packs, dict):
                partial_packs = [partial_packs]

            all_packs = list(partial_packs)

            if client_id in self.models:
                site_name = helpers.find_site_of_node(self.topology, client_id)
                t3_start = int(self.site_end_layer.get(site_name, 0))
                t3_end = int(self.total_layers)

                t3_sd = {k: v.detach().cpu() for k, v in self.models[client_id].state_dict().items()}
                all_packs.append({
                    "shard": {"tier": "t3", "start": t3_start, "end": t3_end},
                    "weights": t3_sd
                })

            for pack in all_packs:
                if not isinstance(pack, dict) or "shard" not in pack or "weights" not in pack:
                    logger.warning(f"[AGGR] bad pack from client={client_id}: type={type(pack)}")
                    continue

                sh = pack["shard"]
                tier = str(sh.get("tier"))
                start = int(sh.get("start", -1))
                end = int(sh.get("end", -1))

                if start < 0 or end < 0:
                    logger.warning(f"[AGGR] invalid shard signature from client={client_id}: {sh}")
                    continue

                sd = pack["weights"]
                if not isinstance(sd, dict):
                    logger.warning(f"[AGGR] pack weights is not dict (client={client_id}, sig={(tier, start, end)})")
                    continue

                sig = (tier, start, end)
                if sig not in templates:
                    templates[sig] = [(k, self._to_global_key(k, start)) for k in sd.keys()]

                for k, v in sd.items():
                    gk = self._to_global_key(k, start)

                    if not torch.is_tensor(v):
                        if gk not in first_non_tensor:
                            first_non_tensor[gk] = v
                        continue

                    t = v.detach()
                    if t.is_cuda:
                        t = t.cpu()

                    if not t.is_floating_point():
                        if gk not in first_non_tensor:
                            first_non_tensor[gk] = t
                        continue

                    if gk not in acc:
                        acc[gk] = torch.zeros(t.shape, dtype=torch.float32)
                    acc[gk].add_(t.to(torch.float32), alpha=float(n_samples))
                    wsum[gk] += float(n_samples)

        global_sd = {}
        for gk, tensor in acc.items():
            denom = wsum.get(gk, 0.0)
            global_sd[gk] = tensor.div_(denom) if denom > 0.0 else tensor
        acc.clear()
        global_sd.update(first_non_tensor)

        avg_packs = []
        missing = []
        for sig in sorted(templates, key=lambda x: (x[0], x[1], x[2])):
            tier, start, end = sig
            shard_sd = {}
            for shard_key, gk in templates[sig]:
                if gk in global_sd:
                    shard_sd[shard_key] = global_sd[gk]
                else:
                    missing.append((sig, shard_key))

            if not shard_sd:
                continue

            avg_packs.append({
                "shard": {"tier": tier, "start": start, "end": end},
                "weights": shard_sd
            })

        if missing:
            logger.warning(f"[AGGR] {len(missing)} shard keys had no aggregated value, e.g. {missing[:3]}")

        self.global_averaged_weights = {"shards": avg_packs}
        logger.info(f"[Aggregator] Global aggregation complete: {len(global_sd)} parameters averaged over "
                    f"{len(all_clients)} clients, rebuilt into {len(avg_packs)} shards.")

        # Load aggregated T3 weights into local models
        for client_id in all_clients:
            if client_id not in self.models:
                continue

            site_name = helpers.find_site_of_node(self.topology, client_id)
            t3_start = int(self.site_end_layer.get(site_name, 0))
            t3_end = int(self.total_layers)

            t3_match = None
            for p in avg_packs:
                sh = p["shard"]
                if sh.get("tier") == "t3" and int(sh.get("start")) == t3_start and int(sh.get("end")) == t3_end:
                    t3_match = p
                    break

            if t3_match is not None:
                self.models[client_id].load_state_dict(t3_match["weights"], strict=False)

                # Keep optimizer state (Adam moments) — only refresh LR
                for pg in self.optimizers[client_id].param_groups:
                    pg['lr'] = self.learning_rate

        t_end_agg = time.perf_counter()
        delta_agg = t_end_agg - t_start_agg

        helpers.log_performance_metric(
            node_id=self.node_id,
            client_id=None,
            seed=self.seed,
            run_id=self.now,
            metric_name="delta_agg",
            duration=delta_agg,
            round_idx=r,
            batch_idx=-1,
            req_id=f"global_agg_timestamp_{t_end_agg}",
            delay=None
        )

    def _perform_vision_aggregation(self, r):
        """
        Shard-wise weighted FedAvg for vision models. A vision shard is an
        nn.Sequential whose keys are renumbered from zero and carry no absolute layer
        index, so only packs with the same (tier, start, end) signature are comparable.
        """
        t_start_agg = time.perf_counter()
        logger.info(f"=== Starting Global Aggregation (vision, shard-wise FedAvg) | Round {r} ===")

        if not hasattr(self, "site_partial_weights") or self.site_partial_weights is None:
            self.site_partial_weights = {}

        all_clients = list(self.site_partial_weights.keys())
        shard_bucket = defaultdict(list)

        for client_id in all_clients:
            n_samples = self.iterations_train.get(client_id, 0) * self.batch_size
            if n_samples <= 0:
                logger.warning(f"[AGGR-V] client {client_id} reported 0 samples, ignored.")
                continue

            packs = self.site_partial_weights.get(client_id) or []
            if isinstance(packs, dict):
                packs = [packs]
            packs = list(packs)

            # The residual segment this client left on Tier 3 is averaged with the rest.
            if client_id in self.models:
                site_name = helpers.find_site_of_node(self.topology, client_id)
                packs.append({
                    "shard": {"tier": "t3",
                              "start": int(self.site_end_layer.get(site_name, 0)),
                              "end": int(self.total_layers)},
                    "weights": {k: v.detach().cpu()
                                for k, v in self.models[client_id].state_dict().items()},
                })

            for pack in packs:
                if not isinstance(pack, dict) or "shard" not in pack or "weights" not in pack:
                    logger.warning(f"[AGGR-V] bad pack from client={client_id}: {type(pack)}")
                    continue
                sh = pack["shard"]
                start, end = int(sh.get("start", -1)), int(sh.get("end", -1))
                if start < 0 or end < 0:
                    logger.warning(f"[AGGR-V] invalid signature from client={client_id}: {sh}")
                    continue
                sd = pack["weights"]
                if not isinstance(sd, dict):
                    logger.warning(f"[AGGR-V] weights is not a dict (client={client_id}, {sh})")
                    continue
                shard_bucket[(str(sh.get("tier")), start, end)].append((sd, int(n_samples)))

        avg_packs = []
        for sig in sorted(shard_bucket):
            updates = shard_bucket[sig]
            total = sum(n for _, n in updates if n > 0)
            if total <= 0:
                continue
            shard_sd = self._fed_avg_weighted(updates, total)
            if not shard_sd:
                continue
            tier, start, end = sig
            avg_packs.append({"shard": {"tier": tier, "start": start, "end": end},
                              "weights": shard_sd})

        self.global_averaged_weights = {"shards": avg_packs}
        logger.info(f"[AGGR-V] {len(avg_packs)} shards averaged over {len(all_clients)} clients.")

        # Tier 3 adopts its own averaged segment and restarts its optimizer.
        for client_id in all_clients:
            if client_id not in self.models:
                continue
            site_name = helpers.find_site_of_node(self.topology, client_id)
            t3_sig = ("t3", int(self.site_end_layer.get(site_name, 0)), int(self.total_layers))
            match = next((p for p in avg_packs
                          if (str(p["shard"]["tier"]), int(p["shard"]["start"]),
                              int(p["shard"]["end"])) == t3_sig), None)
            if match is None:
                continue
            self.models[client_id].load_state_dict(match["weights"], strict=False)
            self.optimizers[client_id] = optim.SGD(self.models[client_id].parameters(),
                                                   lr=self.learning_rate, momentum=0.9)

        delta_agg = time.perf_counter() - t_start_agg
        helpers.log_performance_metric(
            node_id=self.node_id,
            client_id=None,
            seed=self.seed,
            run_id=self.now,
            metric_name="delta_agg",
            duration=delta_agg,
            round_idx=r,
            batch_idx=-1,
            req_id=f"vision_agg_timestamp_{time.perf_counter()}",
            delay=None
        )

    def _perform_global_aggregation_old_v1(self, r):
        """
        Aggregation used up to 2026-09-17, kept for comparison. Averages only packs whose
        (tier, start, end) signature matches.
        """
        t_start_agg = time.perf_counter()
        logger.info(f"=== Starting Global Aggregation (Shard-wise Weighted FedAvg, v1) | Round {r} ===")

        if not hasattr(self, "site_partial_weights") or self.site_partial_weights is None:
            self.site_partial_weights = {}

        all_clients = list(self.site_partial_weights.keys())
        all_shard_sigs = set()
        shard_bucket = defaultdict(list)

        for client_id in all_clients:
            n_iters = self.iterations_train.get(client_id, 0)
            n_samples = n_iters * self.batch_size
            if n_samples <= 0:
                logger.warning(f"Client {client_id} has 0 samples. Ignored.")
                continue

            partial_packs = self.site_partial_weights.get(client_id, [])
            if partial_packs is None:
                partial_packs = []
            elif isinstance(partial_packs, dict):
                partial_packs = [partial_packs]

            all_packs = list(partial_packs)

            if client_id in self.models:
                site_name = helpers.find_site_of_node(self.topology, client_id)
                t3_start = int(self.site_end_layer.get(site_name, 0))
                t3_end = int(self.total_layers)

                t3_sd = {k: v.detach().cpu() for k, v in self.models[client_id].state_dict().items()}
                all_packs.append({
                    "shard": {"tier": "t3", "start": t3_start, "end": t3_end},
                    "weights": t3_sd
                })

            for pack in all_packs:
                if not isinstance(pack, dict) or "shard" not in pack or "weights" not in pack:
                    logger.warning(f"[AGGR] bad pack from client={client_id}: type={type(pack)}")
                    continue

                sh = pack["shard"]
                tier = str(sh.get("tier"))
                start = int(sh.get("start", -1))
                end = int(sh.get("end", -1))

                if start < 0 or end < 0:
                    logger.warning(f"[AGGR] invalid shard signature from client={client_id}: {sh}")
                    continue

                sig = (tier, start, end)
                all_shard_sigs.add(sig)

                sd = pack["weights"]
                if not isinstance(sd, dict):
                    logger.warning(f"[AGGR] pack weights is not dict (client={client_id}, sig={sig})")
                    continue

                shard_bucket[sig].append((sd, int(n_samples)))

        avg_packs = []
        for (tier, start, end) in sorted(all_shard_sigs, key=lambda x: (x[0], x[1], x[2])):
            sig = (tier, start, end)
            updates = shard_bucket.get(sig, [])
            if not updates:
                continue

            total_samples_sig = sum(ns for _, ns in updates if ns and ns > 0)
            if total_samples_sig <= 0:
                continue

            shard_sd = self._fed_avg_weighted(updates, total_samples_sig)
            if not shard_sd:
                continue

            avg_packs.append({
                "shard": {"tier": tier, "start": start, "end": end},
                "weights": shard_sd
            })

        self.global_averaged_weights = {"shards": avg_packs}
        logger.info(f"[Aggregator] Global aggregation complete with {len(avg_packs)} shards.")

        # Load aggregated T3 weights into local models
        for client_id in all_clients:
            if client_id not in self.models:
                continue

            site_name = helpers.find_site_of_node(self.topology, client_id)
            t3_start = int(self.site_end_layer.get(site_name, 0))
            t3_end = int(self.total_layers)

            t3_match = None
            for p in avg_packs:
                sh = p["shard"]
                if sh.get("tier") == "t3" and int(sh.get("start")) == t3_start and int(sh.get("end")) == t3_end:
                    t3_match = p
                    break

            if t3_match is not None:
                self.models[client_id].load_state_dict(t3_match["weights"], strict=False)

                if self.model_type == "llm":
                    # Keep optimizer state (Adam moments) — only refresh LR
                    for pg in self.optimizers[client_id].param_groups:
                        pg['lr'] = self.learning_rate
                else:
                    self.optimizers[client_id] = optim.SGD(self.models[client_id].parameters(),
                                                           lr=self.learning_rate,
                                                           momentum=0.9)

        t_end_agg = time.perf_counter()
        delta_agg = t_end_agg - t_start_agg

        helpers.log_performance_metric(
            node_id=self.node_id,
            client_id=None,
            seed=self.seed,
            run_id=self.now,
            metric_name="delta_agg",
            duration=delta_agg,
            round_idx=r,
            batch_idx=-1,
            req_id=f"global_agg_timestamp_{t_end_agg}",
            delay=None
        )

    def _fed_avg_weighted(self, weighted_updates, total_samples):

        if total_samples <= 0:
            raise ValueError("total_samples must be > 0")

        acc = {}
        wsum = defaultdict(float)
        first_non_tensor = {}  # buffers/ints: keep first seen

        for sd, n_samples in weighted_updates:
            if n_samples <= 0:
                continue
            w = float(n_samples) / float(total_samples)

            for k, v in sd.items():
                if not torch.is_tensor(v):
                    if k not in first_non_tensor:
                        first_non_tensor[k] = v
                    continue

                t = v.detach()
                # Ensure CPU to avoid accidentally holding VRAM
                if t.is_cuda:
                    t = t.cpu()

                # Accumulate floating-point tensors in FP32
                if t.is_floating_point():
                    t32 = t.float()
                    if k in acc:
                        acc[k].add_(t32, alpha=w)
                    else:
                        acc[k] = t32.mul(w)
                    wsum[k] += w
                else:
                    # ints/bools: keep first seen
                    if k not in first_non_tensor:
                        first_non_tensor[k] = t

        # Normalize by per-key weights (important if some keys are missing from some clients)
        out = {}
        for k, v in acc.items():
            denom = wsum.get(k, 0.0)
            if denom > 0.0:
                out[k] = v.div(denom)
            else:
                out[k] = v

        out.update(first_non_tensor)

        return out

    def _fed_avg_simple(self, weighted_updates, total_samples):

        if total_samples <= 0:
            raise ValueError("total_samples must be > 0")

        acc = {}
        count = defaultdict(int)
        first_non_tensor = {}
        num_contributors = 0

        for sd, n_samples in weighted_updates:
            if n_samples <= 0:
                continue

            num_contributors += 1

            for k, v in sd.items():
                if not torch.is_tensor(v):
                    if k not in first_non_tensor:
                        first_non_tensor[k] = v
                    continue

                t = v.detach()
                if t.is_cuda:
                    t = t.cpu()

                if t.is_floating_point():
                    t32 = t.float()
                    if k in acc:
                        acc[k] += t32
                    else:
                        acc[k] = t32.clone()
                    count[k] += 1
                else:
                    if k not in first_non_tensor:
                        first_non_tensor[k] = t

        out = {}
        for k, v in acc.items():
            n = count.get(k, 0)
            if n > 0:
                out[k] = v.div(float(n))
            else:
                out[k] = v

        out.update(first_non_tensor)

        logger.info(f"[FedAvg] Averaged {len(acc)} tensor keys from {num_contributors} clients (simple averaging)")

        return out

    def _distribute_aggregated_weights(self):
        logger.info("=== Distributing Global Weights to Clusters ===")

        # Filter out T3 shards (they stay on Orchestrator)
        non_t3_shards = [p for p in self.global_averaged_weights.get("shards", [])
                         if p["shard"].get("tier") != "t3"]

        for site_key, sock in self.connected_clusters.items():
            msg_payload = {"shards": non_t3_shards, "version": "global"}
            logger.info(f"Sending GLOBAL WEIGHTS to {site_key}.")
            self.send_msg(sock, ["UPDATE_GLOBAL_WEIGHTS", msg_payload])

    def _perform_global_evaluation(self, current_round, training_time):
        logger.info("=== Starting Global Evaluation Phase (s1_t1_1 only) ===")

        eval_start_time = time.perf_counter()
        self.eval_metrics_buffer = {}

        # Determine which site hosts s1_t1_1
        eval_client = "s1_t1_1"
        eval_site = helpers.find_site_of_node(self.topology, eval_client)

        # Send START_EVALUATION to evaluating site, SKIP_EVALUATION to others
        for site_name, sock in self.connected_clusters_data.items():
            if site_name == eval_site:
                self.send_msg(sock, ["START_EVALUATION", {"round": current_round, "eval_client": eval_client}])
            else:
                self.send_msg(sock, ["SKIP_EVALUATION", {"round": current_round}])

        # Only spawn evaluation thread for s1_t1_1
        eval_sock = self.client_train_sockets.get(eval_client)
        if eval_sock:
            t = threading.Thread(
                target=self._handle_evaluation_stream,
                args=(eval_client, eval_sock, current_round),
                name=f"Eval-{eval_client}"
            )
            t.daemon = True
            t.start()
            t.join()

        eval_duration = time.perf_counter() - eval_start_time
        logger.info(f"[Eval] Phase finished in {eval_duration:.2f}s.")

        self._save_metrics(current_round, training_time, eval_duration)

    def _handle_evaluation_stream(self, client_id, sock, r):
        """
        Thread that processes the evaluation stream from a client, aggregating loss_sum,
        correct and num_tokens.
        """
        model_shard = self.models.get(client_id, None)
        if model_shard is not None:
            model_shard.eval()

        acc = helpers.StreamingEvalAccumulator()
        iterations_eval = self.iterations_eval[client_id]
        for i in tqdm.tqdm(range(iterations_eval), desc=f"[{self.node_id} | {client_id} | Round: {r}]"):

            # msg = self.recv_msg(sock, "FORWARD_EVAL")
            msg = self.recv_msg(sock)
            command = msg[0]
            payload = msg[1]
            # payload = msg[1]

            if command == "EVAL_METRICS":
                acc.update(payload['loss_sum'], payload['correct'], payload['n_tokens'])

            elif command == "FORWARD_EVAL":

                inputs = payload["data"].to(self.device)
                labels = payload["labels"].to(self.device)

                mask_raw = payload.get("attention_mask")
                mask = mask_raw.to(self.device) if mask_raw is not None else None

                pos_raw = payload.get("position_ids")
                pos_ids = pos_raw.to(self.device) if pos_raw is not None else None

                req_id = payload.get("req_id", None)
                del payload

                t_start_eval = time.perf_counter()

                with torch.no_grad():
                    if model_shard is not None:
                        if self.model_type == "llm":
                            outputs = model_shard(inputs,
                                                  attention_mask=mask,
                                                  position_ids=pos_ids)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                        else:
                            outputs = model_shard(inputs)
                    else:
                        outputs = inputs

                    loss_sum, correct, n_tokens = self._calculate_eval_metrics(outputs, labels)
                    acc.update(loss_sum, correct, n_tokens)

                t_comp = time.perf_counter() - t_start_eval

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=client_id,
                    seed=self.seed,
                    run_id=self.now,
                    metric_name="T_eval",
                    duration=t_comp,
                    round_idx=-1,
                    batch_idx=-1,
                    req_id=req_id,
                    delay=None
                )
                del inputs, labels, mask, pos_ids, outputs
                if i % 10 == 0:
                    gc.collect()

        gc.collect()
        summary = acc.finalize(self.model_type)

        self.eval_metrics_buffer[client_id] = {"loss_sum": summary["loss_sum"],
                                               "correct": summary["correct"],
                                               "num_tokens": summary["num_tokens"],
                                               "loss": summary["avg_loss"],
                                               "acc": summary.get("next_token_acc", summary.get("avg_acc", 0.0)),
                                               "ppl": summary["avg_ppl"]
                                               }

        if model_shard is not None:
            model_shard.train()

    def _calculate_eval_metrics(self, output, labels):
        """
        Returns the per-batch sums (loss_sum, correct, num_tokens). num_tokens counts valid
        tokens for an LLM and samples for vision.
        """
        loss_sum, correct, num_tokens = helpers.compute_eval_batch_metrics(
            model_type=self.model_type,
            output=output,
            labels=labels,
            criterion=self.criterion,
            device=self.device
        )
        return loss_sum, correct, num_tokens

    def _save_metrics(self, r, train_time, eval_time):
        """
        Saves evaluation metrics from s1_t1_1 (the single evaluator).
        Since all clients share the same global model, one evaluation is sufficient.
        """
        eval_client = "s1_t1_1"
        rep_metrics = self.eval_metrics_buffer.get(eval_client)

        if rep_metrics is None:
            logger.warning(f"[Eval] No valid metrics found for {eval_client}")
            return

        tokens = int(rep_metrics.get("num_tokens", 0))
        if tokens <= 0:
            logger.warning(f"[Eval] {eval_client} has zero tokens.")
            return

        loss_sum = float(rep_metrics.get("loss_sum", float(rep_metrics.get("loss", 0.0)) * tokens))
        correct = int(rep_metrics.get("correct", round(float(rep_metrics.get("acc", 0.0)) * tokens)))

        avg_loss = loss_sum / tokens
        avg_acc = correct / tokens

        if self.model_type == "llm":
            avg_ppl = helpers.safe_exp(avg_loss)
        else:
            avg_ppl = 0.0

        global_metrics = {
            "avg_loss": avg_loss,
            "avg_ppl": avg_ppl,
            "next_token_acc": avg_acc,
            "avg_acc": avg_acc,
            "num_tokens": tokens,
        }

        helpers.log_evaluation_summary(
            run_id=self.now,
            round_idx=r,
            model_type=self.model_type,
            group_id=0,
            clients_list=[eval_client],
            metrics=global_metrics,
            train_time=train_time,
            eval_time=eval_time,
            seed=self.seed,
            alpha=self.alpha,
        )

        logger.info(
            f"[Metrics][Global] round={r}"
            f" | avg_loss={avg_loss:.6f}"
            f" | avg_ppl={avg_ppl:.4f}"
            f" | num_tokens={tokens}"
        )

        logger.info(f"[Metrics] Eval saved via helper (RunID: {self.now})")

    def _collect_site_weights(self, round_idx: int, fp16: bool = True):
        buf = {}
        req_id = f"global-r{round_idx}"

        # Request weights from all sites
        for site_name, sock in self.connected_clusters.items():

            self.send_msg(sock, ["REQUEST_SITE_WEIGHTS", {
                "round": round_idx,
                "req_id": req_id,
                "fp16": fp16
            }])

        # Receive responses
        for site_name, sock in self.connected_clusters.items():

            msg = self.recv_msg(sock, "SITE_WEIGHTS_RESPONSE")

            if not msg:
                logger.warning(f"[WEIGHTS][ORCH] no response from site={site_name}")
                continue

            payload = msg[1]
            if payload.get("req_id") != req_id:
                logger.warning(f"[WEIGHTS][ORCH] req_id mismatch from site={site_name}: "
                               f"got={payload.get('req_id')} expected={req_id}")
                continue

            site_map = payload.get("weights_by_client", {})
            for cid, packs in site_map.items():
                if packs is None:
                    packs = []
                elif isinstance(packs, dict):
                    packs = [packs]
                buf[cid] = packs

            logger.info(f"[WEIGHTS][ORCH] collected from site={site_name}: clients={len(site_map)}")

        self.site_partial_weights = buf  # cid -> [WeightPack, ...]
        return buf

    def _calculate_loss(self, output, labels):
        """
        Computes the loss. For LLMs, applies causal shift to logits/labels.
        """

        if self.model_type == 'vision':
            return self.criterion(output, labels)

        elif self.model_type == 'llm':
            # Shift logits for Causal LM
            logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            return self.criterion(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1)
            )

        return 0.0

    def _client_requires_t3(self, client_id: str) -> bool:
        """
        True  -> Tier 3 has layers to process (STRAGGLER / OFFLOAD)
        False -> STRONG/LOCAL_FINISH (Tier 3 does not process layers; metrics only)
        """
        line = self.global_allocation.get(client_id)
        if line is None:
            # Safe fallback: assume offload needed
            return True
        return getattr(line, "tier3_target", None) != "LOCAL_FINISH"

    def _save_experiment_configuration(self):
        """
        Collects and persists all metadata for the current experiment.
        Includes: Hyperparameters, Topology, Hardware, Splits, and Strong/Straggler classification.
        """
        logger.info("=== Saving Experiment Configuration Snapshot ===")

        # 1. Cluster Classification (Strong vs Straggler)
        cluster_classifications = {}
        allocations_summary = {}

        for node_id, alloc_line in self.global_allocation.items():
            site_name = helpers.find_site_of_node(self.topology, node_id)

            # should_offload=True -> Straggler/Offloader; False -> Strong/Local-Finish
            is_offloader = self.site_should_offload.get(site_name, True)
            classification = "STRAGGLER" if is_offloader else "STRONG"

            if site_name not in cluster_classifications:
                cluster_classifications[site_name] = classification

            allocations_summary[node_id] = {
                "site": site_name,
                "tier1_layers": alloc_line.tier1_split_point,
                "tier2_distribution": alloc_line.tier2_distribution,  # Dict {server_id: layers}
                "tier3_target": alloc_line.tier3_target,
                "classification": classification
            }

        # 2. Hardware Profiles (detected/used)
        hardware_profiles = {}

        all_known_nodes = set(self.profiles.keys()) | set(self.gpus_workers.keys())

        for nid in all_known_nodes:
            profile = self.profiles.get(nid)
            ram = profile.ram if profile else 0.0
            flops = profile.flops if profile else 0.0

            real_gpu_name = self.gpus_workers.get(nid, "Not Reported")

            hardware_profiles[nid] = {
                "ram_gb": ram,
                "flops": flops,
                "gpu_name_detected": real_gpu_name
            }

        # 3. Build master metadata object
        metadata = {
            "meta": {
                "run_id": self.now,
                "timestamp": datetime.datetime.now().isoformat(),
                "seed": self.seed,
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device": str(self.device),
                "initial_system_snapshot": self.initial_snapshot
            },
            "training_phase":{
                "clip_max_norm": self.clip_max_norm,
                "rounds": self.global_rounds,
                "iterations_train": self.iterations_train
            },
            "model": {
                "name": self.model_name,
                "type": self.model_type,
                "total_layers": self.total_layers,
                "alpha": self.alpha,
                "input_embeddings_cust_gib": self.input_embeddings_cust_gib,
                "transformers_block_standard_cust_gib": self.transf_block_standard_cust_gib,
                "transformers_block_checkpointed_cust_gib": self.transf_block_checkpointed_cust_gib,
                "lm_head_cost_gib": self.lm_head_cost_gib,
                "logits_tensor_cost_gib": self.logits_tensor_cost_gib,
                "seq_len": self.seq_len,
                "batch_size": self.batch_size,

            },
            "hyperparameters": self.hyperparameters,
            "topology_summary": {
                "sites": list(self.topology['sites'].keys()),
                "total_clients": len(self.client_train_sockets) + len(self.connected_clusters_data)  # Estimativa
            },
            "experiment_results": {
                "cluster_classifications": cluster_classifications,
                "detailed_allocations": allocations_summary
            },
            "node_hardware_profiles": hardware_profiles
        }

        helpers.save_experiment_metadata(self.node_id, self.now, metadata)


