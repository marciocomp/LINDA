# @author: Marcio Lopes
import sys
import torch

sys.path.append('../../../../')
from frameworkLINDA.tier_3.orchestrator import GlobalOrchestrator
from utils.linda_logger import logger
from utils.json_helper import load_json


class ConfigLeader:
    """
    Builder pattern for GlobalOrchestrator.
    """

    def __init__(self,
                 hyperparams=None,
                 topology_json_path="../network_config.json",
                 # hw_specs_json_path="../hardware_specs.json",
                 links_json_path="../network_links.json"
                 ):

        self.hyperparams = hyperparams if hyperparams else {}
        self.seed = self.hyperparams.get("seed")
        self.site_name = self.hyperparams.get("site", "site1")
        self.model_name = self.hyperparams.get("model_name")
        self.model_type = self.hyperparams.get("model_type")
        self.alpha = self.hyperparams.get("alpha")
        self.straggler = self.hyperparams.get("straggler")
        self.global_rounds = self.hyperparams.get("rounds")

        self.topology = load_json(topology_json_path)
        # self.hardware_specs = load_json(hw_specs_json_path)

        self.network_links = load_json(links_json_path)

        if self.site_name not in self.topology['sites']:
            self.site_name = list(self.topology['sites'].keys())[0]

        self.local_config = self.topology['sites'][self.site_name]['tier3']['leader']

        self.global_rounds = self.hyperparams.get("rounds")

    def build_orchestrator(self):
        """
        System construction flow.
        """
        orchestrator = GlobalOrchestrator(
            seed=self.seed,
            node_id=self.local_config['id'],
            ip_address= self.local_config['ip'],
            port=self.local_config['port'][0],
            data_port=self.local_config['port'][1],
            straggler=self.straggler,
            hyperparameters=self.hyperparams,
            topology=self.topology,
            # hardware_specs=self.hardware_specs,
            network_links=self.network_links,
            model_name=self.model_name,
            model_type=self.model_type,
            alpha=self.alpha,
            global_rounds= self.global_rounds,
            device=self.local_config["device"]
        )
        logger.info("============================================")
        logger.info("====== SYSTEM INITIALIZATION STARTING ======")
        logger.info("============================================")

        orchestrator.reallocation_enabled = bool(self.hyperparams.get("realloc", False))
        orchestrator.realloc_min_streak = int(self.hyperparams.get("realloc_min_rounds", 1))
        logger.info(f"Runtime re-allocation: "
                    f"{'ON' if orchestrator.reallocation_enabled else 'OFF'} "
                    f"(min rounds: {orchestrator.realloc_min_streak})")

        return orchestrator

    def discover_and_allocate(self, orchestrator):
        """Algorithms 1 and 2: read the infrastructure, then place the model on it."""

        # 2. Discovery (Algorithm 1)
        resourceVector = orchestrator.request_site_profiles()
        resourceVector = orchestrator.run_profiling()
        graphN = orchestrator.build_network_graph()

        # 3. Allocation Decision (Algorithm 2)
        is_train_possible, initial_matrix = orchestrator.generate_initial_allocation()

        if is_train_possible:

            # 4. Configuration Injection
            start, end = self.get_split_points(initial_matrix, orchestrator.node_id)

            orchestrator.start_layer = start
            orchestrator.end_layer = end

            return True
        else:
            return False

    def get_split_points(self, initial_matrix, leader_id):
        """
        Computes the global minimum entry point for the Orchestrator.
        If a Straggler site sends from layer 2 and a Strong site from layer 16,
        the Orchestrator must load from layer 2 onward.
        """
        if not initial_matrix:
            return 0, None

        min_start_layer = float('inf')

        for node_id, allocation_line in initial_matrix.items():
            # Tier 1 split point
            current_layer = allocation_line.tier1_split_point

            # Sum layers processed by Tier 2 (excluding the Leader itself)
            remote_t2_layers = 0
            for s_id, layers in allocation_line.tier2_distribution.items():
                if s_id != leader_id:
                    remote_t2_layers += layers

            # Arrival point at the Orchestrator
            arrival_at_leader = current_layer + remote_t2_layers

            if arrival_at_leader < min_start_layer:
                min_start_layer = arrival_at_leader

        if min_start_layer == float('inf'):
            min_start_layer = 0

        # Returns start_layer and None (None = until the end)
        return int(min_start_layer), None