# @author: Marcio Lopes
import sys
import os

sys.path.append('../../../../')

from frameworkLINDA.tier_2.cluster_manager import ClusterManager
from utils.linda_logger import logger
from utils.json_helper import load_json


class ConfigClusterManager:
    """
    Configuration Builder for Tier 2 (Cluster Manager).
    Renamed from ConfigWorker to reflect its role better.

    Responsibilities:
    1. Read Network Topology.
    2. Identify Local Configuration (My IP/Port) based on Node ID.
    3. Identify Orchestrator Configuration (Target IP/Port).
    4. Instantiate frameworkLINDA.ClusterManager.
    """

    def __init__(self,
                 site_name,
                 node_id,
                 topology_json_path="../network_config.json"):
        """
        Args:
            site_name: The ID of the site this manager belongs to (e.g., 'site2').
            node_id: The specific ID of this node (e.g., 's2_t2_1').
            topology_json_path: Relative path to topology JSON.
        """
        self.site_name = site_name
        self.node_id = node_id
        # 1. Load Topology
        self.topology = load_json(topology_json_path)
        # 2. Get Orchestrator Info (Always at Site 1 - Tier 3)
        leader_config = self.topology['sites']['site1']['tier3']['leader']
        self.orchestrator_ports = leader_config['port']

        # 3. Get Local Info (My Tier 2 Server)
        local_tier2 = self.topology['sites'][self.site_name]['tier2']

        # Find config matching this node's ID
        self.local_config = None

        for key, conf in local_tier2.items():
            if conf['id'] == self.node_id:
                self.local_config = conf
                break

        # Fallback: if ID not found, use first server
        if self.local_config is None:
            logger.warning(
                f"Node ID {self.node_id} not found in topology for {self.site_name}. Defaulting to first server.")
            first_server_key = next(iter(local_tier2))
            self.local_config = local_tier2[first_server_key]
            self.node_id = self.local_config['id']
        self.ip_address = self.local_config['ip']
        self.local_port_list = self.local_config["port"]
        self.device = self.local_config['device']
        self.orchestrator_ip = leader_config['ip']
        logger.info(f"Config loaded for Node: {self.node_id} (Ports: {self.local_port_list})")


    def get_secondary_workers_count(self):
        """
        Counts how many Tier 2 servers exist in this site excluding the Manager.
        Useful to wait for other compute nodes to connect.
        """
        local_tier2 = self.topology['sites'][self.site_name]['tier2']
        # Total servers minus 1 (the manager itself)
        return max(0, len(local_tier2) - 1)

    def build_manager(self):
        """
        Factory method to create the ClusterManager.
        """
        logger.info(f"Building Cluster Manager for {self.site_name} ({self.ip_address})...")
        logger.info(f"Target Orchestrator: {self.orchestrator_ip}:{self.orchestrator_ports}")

        manager = ClusterManager(
            node_id=self.node_id,
            site_name=self.site_name,
            ip_address=self.ip_address,
            local_port_list=self.local_port_list,
            orchestrator_ip=self.orchestrator_ip,
            orchestrator_ports=self.orchestrator_ports,
            topology=self.topology,
            device=self.device
        )



        return manager