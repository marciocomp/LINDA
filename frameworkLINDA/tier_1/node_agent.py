# @author: Marcio Lopes
import random
import socket
import sys
import threading
import time
import uuid
import transformers
import gc
import numpy as np
import psutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import traceback
import tqdm
import datetime
import resource

sys.path.append('../../')
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers
from utils.standardTuples.data_payload import ForwardDataTuple, BackwardGradTuple

from utils.datasets.alpaca import AlpacaDataset, collate_fn
from utils.datasets.data_partitioner import DataPartitioner
from utils.datasets.cifar_10 import Cifar10Dataset
from utils.datasets.imagenet import ImageNetDataset

class NodeAgent(CommunicationModule):
    """
    Tier 1 Component: The Data Source / Edge Device.
    Responsibilities:
    1. Holds the local dataset partition.
    2. Executes the first layers of the DNN (Shard 1).
    3. Offloads intermediate data (Smashed Data) to Tier 2.
    4. Receives Gradients and updates local weights.
    """

    def __init__(self,
                 node_id,
                 ip_address,
                 manager_ip,
                 manager_port_control,
                 manager_port_data,
                 device,
                 data_dir,
                 dataset_name,
                 new_size,
                 proxies=None):

        super().__init__(node_id, ip_address)

        self.proxies = proxies or {}
        self.fedprox_cpu = None
        self.weight_decay = None
        self.global_params = None
        self.num_classes = None
        self.clip_max_norm = None
        self.learning_rate = None
        self._early_eval_signal = None
        self.global_rounds = None
        self.data_dir = data_dir
        self.manager_ip = manager_ip
        self.manager_port_control = manager_port_control
        self.manager_port_data = manager_port_data
        self.manager_sock_control = None
        self.manager_sock_data = None
        self.new_size = new_size
        self.classes = None
        self.snapshot_end = {}
        self.snapshot_start = {}

        self.config = None
        self.split_point = 0
        self.client_training_id = 0
        self.total_clients = 0
        self.seed = None
        self.alpha = 0.5
        self.batch_size = None
        self.model_name = None
        self.model_type = None
        self.num_samples = None
        self.dataset_name = dataset_name


        # if device == "cpu":
        self.device = torch.device(device) # device  # "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        #     self.device = torch.device(device if torch.cuda.is_available() else "cpu")


        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.iterations_train = None
        self.iterations_eval = None
        self.output_norm = None
        self.mu = 0.0

        self.execution_context = {}
        self.context_lock = threading.Lock()
        self.run_id = None
        self.hw_name = helpers.get_local_device_name(self.device)
        self.delay = 0.0

        logger.info(f"Node Agent {node_id} initialized on {self.device}.")
        logger.info(f"Device Name: {self.hw_name}")
        self.initial_snapshot = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Initial System Snapshot : {self.initial_snapshot}")


    def _set_global_seed(self):
        """
        Sets the random seed for reproducibility across all libraries.
        """

        logger.info(f"Setting Global Seed to: {self.seed}")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
           torch.cuda.manual_seed_all(self.seed)

    def connect_to_manager_control(self):
        """Connects to the Tier 2 Cluster Manager."""
        dial = helpers.dial_ip(self.proxies, self.ip, self.manager_ip)
        logger.info(f"Connecting to Manager at {self.manager_ip}:{self.manager_port_control} "
                    f"(dialing {dial}:{self.manager_port_control})...")
        try:
            self.sock_client.connect((dial, self.manager_port_control))
            self.manager_sock_control = self.sock_client
            # hw_name = helpers.get_local_device_name(self.device)
            msg = ["HELLO_DEVICE", [self.node_id, self.hw_name,
                                     helpers.measure_local_resources(self.node_id, self.device)]]
            self.send_msg(self.manager_sock_control, msg)
            logger.info("-> Connected to Manager Control!")
            self.perform_time_sync_handshake(self.manager_sock_control, role="client")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Manager: {e}")
            return False

    def connect_to_manager_data(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((helpers.dial_ip(self.proxies, self.ip, self.manager_ip), self.manager_port_data))
            self.manager_sock_data = sock
            msg = ["HELLO_DEVICE_DATA", self.node_id]
            self.send_msg(self.manager_sock_data, msg)
            logger.info("-> Connected to Manager Data!")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Manager: {e}")
            return False


    def wait_for_start_signal(self):
        """
        Blocks waiting for the start signal from the Cluster Manager.
        Receives hyperparameters and updates the local optimizer.
        """
        logger.info("Waiting for START_TRAINING signal...")

        msg = self.recv_msg(self.manager_sock_control, "START_TRAINING")

        if msg:
            config = msg[1]
            self.global_rounds = config.get("global_rounds", 1)
            self.mu = config.get("mu", 0.0)
            self.fedprox_cpu = config.get("fedprox_cpu", False)
            self.weight_decay = config.get("weight_decay", 0.01)

            logger.info(f"Start Signal Received! Configuration: {config}")

            if self.optimizer:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                logger.info(f"Local Optimizer updated with LR: {self.learning_rate}")

            return True
        return False

    def send_iterations(self):
        logger.info(f"Sending iterations: {self.iterations_train}")
        msg = ["ITERATIONS", self.iterations_train, self.iterations_eval]
        self.send_msg(self.manager_sock_control, msg)

    def wait_for_instructions(self):
        """
        Blocking call. Waits for configuration.
        """
        logger.info("Waiting for instructions...")
        msg = self.recv_msg(self.manager_sock_control, "AGENT_CONFIG")


        if msg:
            self.config = msg[1]
            self.split_point = self.config.get('split_point', 1)
            self.client_training_id = self.config.get('client_training_id', 0)
            self.total_clients = self.config.get('total_clients', 0)  # Fallback
            self.seed = self.config.get('seed', 42)
            self.alpha = self.config.get('alpha', 0.5)
            self.batch_size = self.config.get('batch_size', 4)
            self.model_name = self.config.get('model_name')
            self.model_type = self.config.get('model_type')
            self.run_id = self.config['run_id']
            self.enable_net_metrics(self.run_id)
            self.learning_rate =self.config['learning_rate']
            self._set_global_seed()

            logger.info(f"Config Received. Split: {self.split_point} | ID: {self.client_training_id}")

            if self.model_type == "llm":
                self.clip_max_norm = 1.0
            else:
                self.clip_max_norm = 1.0

            self._prepare_data()
            self.send_iterations()
            # self._initialize_model()
            self._save_experiment_configuration()

            return True
        return False



    def _prepare_data(self):
        """Loads partitioned dataset with GLOBAL fixed eval (10%) + per-client train split.

        Pipeline (LLM):
          1) Load global dataset indices
          2) Reduce universe by new_size (global)
          3) Split 10% global eval indices (fixed for all clients)
          4) Partition remaining train pool among clients (non-IID)
        """
        logger.info("Preparing Dataset...")
        try:
            collate = collate_fn if self.model_type == "llm" else None

            if self.model_type == "llm":
                self.dataset_name = "tatsu-lab/alpaca"

                # ------------------------------------------------------------
                # 0) Build global reduced universe + global eval + train pool
                # ------------------------------------------------------------
                eval_indices, train_pool_indices, universe_indices = DataPartitioner.get_global_eval_and_train_pool(
                    dataset_name=self.dataset_name,
                    new_size=self.new_size,  # applied BEFORE everything
                    eval_fraction=0.10,  # 10% global fixed
                    seed=self.seed,
                    cache_dir=self.data_dir
                )

                self.global_eval_indices = eval_indices
                self.global_universe_indices = universe_indices
                self.global_train_pool_indices = train_pool_indices

                logger.info(
                    f"[DATA][LLM] Universe(reduced)={len(universe_indices)} | "
                    f"GlobalEval(10%)={len(eval_indices)} | TrainPool={len(train_pool_indices)}"
                )

                # ------------------------------------------------------------
                # 1) Partition ONLY the train pool among clients (non-IID)
                # ------------------------------------------------------------
                train_indices = DataPartitioner.get_partition_indices(
                    dataset_name=self.dataset_name,
                    total_clients=self.total_clients,
                    partition_id=self.client_training_id,
                    method= "content_dirichlet_balanced", #"content_non_iid",  # ou "iid" / "quantity_skew"
                    alpha=self.alpha,
                    seed=self.seed,
                    cache_dir=self.data_dir,
                    pool_indices=train_pool_indices
                )

                logger.info(
                    f"[DATA][LLM] Client={self.client_training_id}/{self.total_clients} "
                    f"TrainSamples={len(train_indices)} | EvalSamples(shared)={len(eval_indices)}"
                )


                # ------------------------------------------------------------
                # 2) Instantiate datasets (train per-client, eval shared)
                # ------------------------------------------------------------
                self.train_dataset = AlpacaDataset(
                    tokenizer_name=self.model_name,
                    split="train",
                    indices=train_indices,
                    cache_dir=self.data_dir
                )

                self.val_dataset = AlpacaDataset(
                    tokenizer_name=self.model_name,
                    split="train",  # Alpaca only has a train split
                    indices=eval_indices,
                    cache_dir=self.data_dir
                )

                # ------------------------------------------------------------
                # 3) DataLoaders
                # ------------------------------------------------------------
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=collate,
                    drop_last=True,
                    pin_memory=True if self.device.type == "cuda" else False,
                )

                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    drop_last=False,
                    pin_memory=True if self.device.type == "cuda" else False,
                )

                self.num_samples = len(self.train_dataset)
                self.iterations_train = len(self.train_loader)
                self.iterations_eval = len(self.val_loader)


            else:

                if self.dataset_name == "imagenet":

                    self.dataset_name = "ImageNet"
                    self.num_classes = 1000

                    self.train_dataset = ImageNetDataset(
                        root=self.data_dir,
                        train=True,
                        client_id=self.client_training_id,
                        total_clients=self.total_clients,
                        alpha=self.alpha,
                        seed=self.seed,
                        new_size=self.new_size,
                        image_size=self.config.get("image_size", 224)
                    )

                    self.val_dataset = ImageNetDataset(
                        root=self.data_dir,
                        train=False,
                        new_size=self.new_size,
                        image_size=self.config.get("image_size", 224)
                    )

                else:
                    self.dataset_name = "CIFAR10"
                    self.num_classes = 10
                    self.train_dataset = Cifar10Dataset(
                        root=self.data_dir,
                        train=True,
                        client_id=self.client_training_id,
                        total_clients=self.total_clients,
                        alpha=self.alpha,
                        seed=self.seed,
                        new_size=self.new_size
                    )
                    self.val_dataset = Cifar10Dataset(
                        root=self.data_dir,
                        train=False,
                        new_size=self.new_size
                    )

                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=collate,
                    drop_last=True,
                    num_workers=4 if self.dataset_name == "ImageNet" else 0,  # I/O bound
                    pin_memory=True if self.device.type == "cuda" else False,
                )

                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    drop_last=False,
                    num_workers=4 if self.dataset_name == "ImageNet" else 0,
                    pin_memory=True if self.device.type == "cuda" else False,
                )

                self.num_samples = len(self.train_dataset)
                self.iterations_train = len(self.train_loader)
                self.iterations_eval = len(self.val_loader)

            logger.info(
                f"Datasets Ready.\n"
                f"\t\tName: {self.model_name} | Type: {self.model_type}\n"
                f"\t\tDataset: {self.dataset_name}\n"
                f"\t\tTrain Samples: {len(self.train_dataset)} | "
                f"Eval Samples: {len(self.val_dataset)} \n"
                f"\t\tTrain Batches: {len(self.train_loader)} | "
                f"Eval Batches: {len(self.val_loader)} \n"
                f"\t\tIterations: {self.iterations_train}"
            )

        except Exception as e:
            logger.error(f"Data Prep Error: {e}")
            raise

    def _initialize_model(self):
        """Loads the first shard of the model."""
        logger.info(f"Initializing Model Shard (0-{self.split_point})...")
        self.snapshot_start = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Snapshot BEFORE initialization:\n"
                    f"\t\t\t{self.snapshot_start}")

        self.model = ModelFactory.get_model_shard(
            model_name=self.model_name,
            model_type=self.model_type,
            start_layer=0,
            end_layer=self.split_point,
            is_first=True,
            is_last=False,
            device=self.device,
            hf_token=HF_TOKEN,
            num_classes=self.num_classes,
        )

        if self.model_type == "llm":
            self.optimizer = optim.AdamW(self.model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.learning_rate, momentum=0.9)

        logger.info("Model Shard Initialized.")

        snapshot_end = helpers.get_local_resource_snapshot(self.device)
        logger.info(f"Snapshot AFTER initialization:\n"
                    f"\t\t\t{snapshot_end}")
        helpers.log_snapshots(node_id=self.node_id,
                              run_id=self.run_id,
                              snapshot_start=self.snapshot_start,
                              snapshot_end=snapshot_end,
                              description="Model_Shard_Load",
                              r=-1,
                              i=-1)

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            helpers.malloc_trim_if_possible()


    def _sync_after_round(self, round_idx):
        logger.info(f"[T1] Round {round_idx}: Entering Sync Phase...")

        # --- STEP 1: Upload (Request) ---
        logger.info(f"[T1] Round {round_idx}: Waiting for REQUEST_LOCAL_WEIGHTS...")
        msg_req = self.recv_msg(self.manager_sock_data)  # strict filter removed for debug

        if msg_req and msg_req[0] == "REQUEST_LOCAL_WEIGHTS":
            self._handle_local_weights_request(msg_req[1], round_idx)
        else:
            logger.error(f"[T1] Sync Error: Expected REQUEST_LOCAL_WEIGHTS, got {msg_req[0] if msg_req else 'None'}")
            return  # Abort sync to avoid deadlock

        # --- STEP 2: Download (Update) ---
        logger.info(f"[T1] Round {round_idx}: Waiting for UPDATE_LOCAL_WEIGHTS...")
        msg_upd = self.recv_msg(self.manager_sock_data)

        if msg_upd and msg_upd[0] == "UPDATE_LOCAL_WEIGHTS":
            logger.info(f"[T1] Round {round_idx}: Global weights received. Updating model...")
            global_weights = msg_upd[1]


            try:
                missing, unexpected = self.model.load_state_dict(global_weights, strict=False)
                if missing:
                    logger.warning(f"[T1] Round {round_idx}: Missing keys in update: {missing[:5]}")
                if unexpected:
                    logger.warning(f"[T1] Round {round_idx}: Unexpected keys in update: {unexpected[:5]}")

                if self.model_type == "llm":
                    # Keep optimizer state (Adam moments) — only refresh LR
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.learning_rate
                else:
                    self.optimizer = optim.SGD(self.model.parameters(),
                                               lr=self.learning_rate,
                                               momentum=0.9)

                logger.info(f"[{self.node_id}] Round {round_idx}: Model updated successfully.")
            except Exception as e:
                logger.error(f"[{self.node_id}] Error loading global weights: {e}")
                raise



        elif msg_upd and msg_upd[0] == "START_EVALUATION":
            # Recovery: the Manager skipped the update, so the eval signal arrived in its place.
            logger.error(f"[T1] CRITICAL: Received START_EVALUATION instead of UPDATE_LOCAL_WEIGHTS. Skipping update.")

            # The eval signal was consumed here, so flag it for wait_start_evaluation.
            self._early_eval_signal = msg_upd  # Save for use in wait_start_evaluation

        else:
            cmd = msg_upd[0] if msg_upd else 'None'
            logger.error(f"[T1] Sync Error: Expected UPDATE_LOCAL_WEIGHTS, got {cmd}")

    def _make_pack_t1(self, fp16: bool) -> dict:
        start = 0
        end = int(self.split_point)
        sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        if fp16:
            sd = helpers.cast_sd_fp16(sd)
        return {"shard": {"tier": "t1", "start": start, "end": end}, "weights": sd}

    def _handle_local_weights_request(self, payload, round_idx):
        """
        Packs local weights in the format expected by the ClusterManager.
        Corrected structure: {'req_id': ..., 'pack': {'weights': ..., 'samples': ...}}
        """
        req_id = payload.get("req_id", f"global-r{round_idx}")

        weights = self.model.state_dict()

        fp16 = bool(payload.get("fp16", False))

        pack_data = self._make_pack_t1(fp16)

        resp_payload = {
            "req_id": req_id,
            "round": round_idx,
            "pack": pack_data
        }

        self.send_msg(self.manager_sock_data, ["LOCAL_WEIGHTS_RESPONSE", resp_payload])
        logger.info(f"[WEIGHTS][T1] Sent weights (fp16) wrapped in 'pack' req_id={req_id}")

    def train(self):
        """
        Main loop managing synchronous forward and backward passes.
        """
        logger.info("=== Starting Split Learning Loop ===")
        # time.sleep(self.client_training_id*4)
        self._initialize_model()
        self.model.train()


        for r in range(self.global_rounds):
            logger.info(f"--- Global Round {r}/{self.global_rounds-1} ---")
            t_start_round = time.perf_counter()

            # --- LR Decay (once per round) ---
            if self.model_type == 'llm':
                if r > 0:
                    self.learning_rate *= 0.9
            elif self.model_type == 'vision':
                if r in (49, 74, 99, 124, 149):
                    self.learning_rate *= 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            logger.info(f"[Round {r}] LR: {self.learning_rate}")

            # FedProx: save global weights before local training
            if self.mu > 0:
                if self.fedprox_cpu:
                    self.global_params = {n: p.detach().cpu() for n, p in self.model.named_parameters()}
                else:
                    self.global_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}

            self.snapshot_start = helpers.get_local_resource_snapshot(self.device)

            for i, batch in enumerate(tqdm.tqdm(self.train_loader, desc=f"[{self.node_id} | Round: {r}]")):
                req_id = str(uuid.uuid4())

                if self.model_type == 'llm':
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    attention_mask = attention_mask.to(self.device, non_blocking=True)

                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)

                else:
                    input_ids, labels = batch
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    attention_mask = None
                    position_ids = None

                # self.optimizer.zero_grad(set_to_none=True)
                self.optimizer.zero_grad()
                t_start_comp = time.perf_counter()

                if self.model_type == 'llm':
                    output = self.model(input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids
                                        )
                else:
                    output = self.model(input_ids)

                helpers.sync_device(self.device)

                t_end_comp = time.perf_counter()
                t_comp = t_end_comp - t_start_comp

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_forward",
                    duration=t_comp,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )
                snapshot_end = helpers.get_local_resource_snapshot(self.device)
                helpers.log_snapshots(node_id=self.node_id,
                                      run_id=self.run_id,
                                      snapshot_start=self.snapshot_start,
                                      snapshot_end=snapshot_end,
                                      r=r,
                                      i=i,
                                      description=f"After_forward")
                t_start_d2h = time.perf_counter()

                data_to_send = output.detach().cpu()

                payload = {
                    'req_id': req_id,
                    'source': self.node_id,
                    'data': data_to_send,
                    'labels': labels.detach().cpu(),
                    "attention_mask": attention_mask.detach().cpu() if attention_mask is not None else None,
                    "position_ids": position_ids.detach().cpu() if position_ids is not None else None,
                }
                t_end_d2h = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
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
                                         client_id=self.node_id, round_idx=r,
                                         hop="T1->T2", payload=payload, batch_idx=i)


                t_start_comm = time.perf_counter()

                self.send_msg(self.manager_sock_data, ["FORWARD_DATA", payload])

                t_end_comm = time.perf_counter()
                t_comm = t_end_comm - t_start_comm
                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comm_forward",
                    duration=t_comm,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )


                # self._wait_and_process_gradients(expected_req_id=req_id,i=i,r=r)
                t_start_wait = time.perf_counter()

                msg = self.recv_msg(self.manager_sock_data, "BACKWARD_DATA")

                t_end_wait = time.perf_counter()

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
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
                # received_req_id = payload.get('req_id', None)

                grad_tensor = payload['grad'].to(self.device, non_blocking=True)
                original_source= payload['original_source']


                if original_source != self.node_id:
                    raise ValueError(f"Original source [{original_source}] "
                                     f"mismatch with node_id [{self.node_id}]")

                grad_tensor = grad_tensor.to(dtype=output.dtype)

                t_start_comp_backward = time.perf_counter()
                t_start_comp_backward_output = t_start_comp_backward

                output.backward(grad_tensor)

                # FedProx proximal gradient
                gp = getattr(self, 'global_params', None)
                if self.mu > 0 and gp is not None:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in gp:
                            param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

                # -------------------------------------------------------------

                helpers.sync_device(self.device)

                t_end_comp_backward_output = time.perf_counter()
                t_comp_backward_output = t_end_comp_backward_output - t_start_comp_backward_output

                t_start_comp_clip = time.perf_counter()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)

                helpers.sync_device(self.device)

                t_end_comp_clip = time.perf_counter()
                t_comp_clip = t_end_comp_clip - t_start_comp_clip

                t_start_comp_optim = time.perf_counter()

                self.optimizer.step()

                helpers.sync_device(self.device)

                t_end_comp_optim = time.perf_counter()
                t_comp_optim = t_end_comp_optim - t_start_comp_optim

                helpers.sync_device(self.device)

                t_end_comp_backward = time.perf_counter()
                t_comp_backward = t_end_comp_backward - t_start_comp_backward


                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_backward.output.backward",
                    duration=t_comp_backward_output,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_backward.clip_grad_norm",
                    duration=t_comp_clip,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_backward.optimizer.step",
                    duration=t_comp_optim,
                    round_idx=r,
                    batch_idx=i,
                    req_id=req_id,
                    delay=None
                )

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_comp_backward",
                    duration=t_comp_backward,
                    round_idx=r,
                    batch_idx=i,
                    req_id= req_id,
                    delay=None
                )

                snapshot_end = helpers.get_local_resource_snapshot(self.device)
                helpers.log_snapshots(node_id=self.node_id,
                                      run_id=self.run_id,
                                      snapshot_start=self.snapshot_start,
                                      snapshot_end=snapshot_end,
                                      r=r,
                                      i=i,
                                      description=f"After_backward")



            t_end_round = time.perf_counter()
            t_round = t_end_round - t_start_round
            helpers.log_performance_metric(
                node_id=self.node_id,
                client_id=self.node_id,
                seed=self.seed,
                run_id=self.run_id,
                metric_name="T_round",
                duration=t_round,
                round_idx=r,
                batch_idx=self.iterations_train,
                req_id=None,
                delay=None
            )

            snapshot = helpers.get_local_resource_snapshot(self.device)
            logger.info(f"Snapshot: {snapshot}")

            # FedProx: free global params snapshot after round
            self.global_params = None

            self._sync_after_round(round_idx=r)

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            else:
                helpers.malloc_trim_if_possible()


            if self.wait_start_evaluation(round_idx=r):
                self.evaluate()
            else:
                logger.info(f"[T1] Round {r}: Evaluation skipped.")

            logger.info("Evaluation done. Returning to wait loop.")

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            else:
                helpers.malloc_trim_if_possible()

        logger.info("Training Finished.")
  
    def _wait_and_process_gradients(self, expected_req_id,i,r):
        """
        Blocks until receiving the gradient matching req_id and applies local backward.
        """

        msg = self.recv_msg(self.manager_sock_data, "BACKWARD_DATA")

        payload = msg[1]
        received_req_id = payload.get('req_id', None)

        grad_tensor = payload['grad'].to(self.device)

        outputs = self.execution_context.pop(received_req_id, None)

        try:
            grad_tensor = grad_tensor.to(dtype=outputs.dtype)
        except Exception:
            pass

        t_start_comp_backward = time.perf_counter()

        outputs.backward(grad_tensor)

        # FedProx proximal gradient
        gp = getattr(self, 'global_params', None)
        if self.mu > 0 and gp is not None:
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in gp:
                    param.grad.data.add_(self.mu * (param.data - gp[name].data.to(param.device)))

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)

        self.optimizer.step()

        helpers.sync_device(self.device)

        t_end_comp_backward = time.perf_counter()
        t_comp_backward = t_end_comp_backward - t_start_comp_backward

        helpers.log_performance_metric(
            node_id=self.node_id,
            client_id=self.node_id,
            seed=self.seed,
            run_id=self.run_id,
            metric_name="T_comp_backward",
            duration=t_comp_backward,
            round_idx=r,
            batch_idx=i,
            req_id=received_req_id,
            delay=None
        )

    def wait_start_evaluation(self, round_idx):
        """
        Waits for the START_EVALUATION signal in a robust way.
        First checks whether the signal was already captured prematurely during sync
        (error recovery), otherwise blocks waiting on the socket.
        """
        # 1. Check recovery flag (set in _sync_after_round)
        if hasattr(self, '_early_eval_signal') and self._early_eval_signal:
            logger.warning(f"[T1] Round {round_idx}: Using early START_EVALUATION signal received during Sync.")
            self._early_eval_signal = None  # Clear flag for the next round
            return True

        # 2. Normal flow: wait on socket
        logger.info(f"[T1] Round {round_idx}: Waiting for START_EVALUATION/SKIP_EVALUATION...")
        msg = self.recv_msg(self.manager_sock_data)

        if msg:
            command = msg[0]
            if command == "SKIP_EVALUATION":
                logger.info(f"[T1] Round {round_idx}: Received SKIP_EVALUATION. Skipping eval.")
                return False
            if command == "START_EVALUATION":
                logger.info(f"[T1] Round {round_idx}: Received Eval Signal.")
                return True

        return False

    def evaluate(self):

        logger.info("=== Starting Evaluation Phase ===")
        self.model.eval()

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            helpers.malloc_trim_if_possible()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(self.val_loader, desc=f"[{self.node_id}]")):
                # time.sleep(5)
                # --- Data Preparation ---
                if self.model_type == 'llm':
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    attention_mask = attention_mask.to(self.device, non_blocking=True)

                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)
                else:
                    # Vision (CIFAR-10)
                    input_ids, labels = batch
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    attention_mask = None
                    position_ids = None

                t_start_eval = time.perf_counter()

                # --- Local Forward (Tier 1) ---
                if self.model_type == 'llm':
                    output = self.model(input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids)
                else:
                    output = self.model(input_ids)

                req_id = str(uuid.uuid4())

                helpers.sync_device(self.device)

                t_end_eval = time.perf_counter()

                t_eval = (t_end_eval - t_start_eval)

                helpers.log_performance_metric(
                    node_id=self.node_id,
                    client_id=self.node_id,
                    seed=self.seed,
                    run_id=self.run_id,
                    metric_name="T_eval",
                    duration=t_eval,
                    round_idx=-1,
                    batch_idx=-1,
                    req_id=req_id,
                    delay=None,
                )

                payload = {
                    'req_id': req_id,
                    'source': self.node_id,
                    'data': output.detach().cpu(),
                    'labels': labels.cpu(),
                    'attention_mask': attention_mask.cpu() if attention_mask is not None else None,
                    'position_ids': position_ids.cpu() if position_ids is not None else None,
                }

                self.send_msg(self.manager_sock_data, ["FORWARD_EVAL", payload])

        logger.info("Evaluation loop finished locally.")
        self.model.train()
    
    def _save_experiment_configuration(self):
        """
        Collects and persists all metadata for the current experiment.
        Includes: Hyperparameters, Topology, Hardware, Splits and Strong/Straggler classification.
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
                "device": self.hw_name,
                "new_size": self.new_size
            },
            "dataset details": {
                "Name": self.dataset_name,
                "Train Samples": len(self.train_dataset),
                "Train Batches": len(self.train_loader),
                "Eval Samples": len(self.val_dataset),
                "Eval Batches": len(self.val_loader),
                "Iterations": self.iterations_train,
            }

        }
        helpers.save_experiment_metadata(self.node_id, self.run_id, metadata)


