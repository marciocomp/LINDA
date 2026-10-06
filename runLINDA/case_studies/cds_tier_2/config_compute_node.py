# @author: Marcio Lopes
import sys

sys.path.append('../../../../')

from frameworkLINDA.tier_2.compute_node import ComputeNode
from utils.linda_logger import logger
from utils.json_helper import load_json
from utils import helpers


class ConfigComputeNode:
    """
    Configuration Builder for Tier 2 (Secondary Compute Node).
    Responsibilities:
    1. Read Network Topology.
    2. Identify Local Configuration (My IP/Port).
    3. Identify Cluster Manager Configuration (My Boss).
    4. Instantiate frameworkLINDA.ComputeNode.
    """

    def __init__(self,
                 site_name,
                 server_id,
                 topology_json_path="../network_config.json"):

        self.site_name = site_name
        self.server_id = server_id
        self.topology = load_json(topology_json_path)
        leader_config = self.topology['sites']['site1']['tier3']['leader']

        self.orchestrator_ports = leader_config['port']

        # 1. Get Local Info

        tier2_conf = self.topology['sites'][self.site_name]['tier2']
        self.local_config = None

        # Find my specific configuration by ID
        for key, conf in tier2_conf.items():
            if conf['id'] == self.server_id:
                self.local_config = conf
                break

        if not self.local_config:
            raise ValueError(f"Server ID {self.server_id} not found in topology for {self.site_name}")

        self.ip_address = self.local_config['ip']
        self.local_port = self.local_config['port'][0]  # Port 0 is for Compute Communication
        self.device = self.local_config['device']


        # 2. Get Cluster Manager Info (Assume First Server in List is Manager)

        manager_key = next(iter(tier2_conf))
        manager_conf = tier2_conf[manager_key]

        self.manager_ip = manager_conf['ip']
        self.manager_port = manager_conf['port'][0]
        self.manager_port_data = manager_conf['port'][3]




    def build_node(self):
        """
        Factory method to create the ComputeNode.
        """
        logger.info(f"Building Compute Node {self.server_id}...")

        node = ComputeNode(
            node_id=self.server_id,
            ip_address=self.ip_address,
            local_port=self.local_port,
            manager_ip=self.manager_ip,
            manager_port=self.manager_port,
            manager_port_data=self.manager_port_data,
            orchestrator_ports=self.orchestrator_ports,
            device=self.device,
            proxies=helpers.load_proxy_map(self.topology)
        )
        return node