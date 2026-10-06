# @author: Marcio Lopes
import sys
sys.path.append('../../../../')

from frameworkLINDA.tier_1.node_agent import NodeAgent
from utils.linda_logger import logger
from utils.json_helper import load_json
from utils import helpers

class ConfigNodeAgent:
    """
    Builder for Tier 1 Node Agent.
    """
    def __init__(self,
                 site_name,
                 node_id,
                 topology_json_path="../network_config.json",
                 new_size=0.1):
        self.site_name = site_name
        self.node_id = node_id
        self.topology = load_json(topology_json_path)
        self.new_size = new_size

        # 1. Find local settings
        try:
            site_tier1 = self.topology['sites'][site_name]['tier1']

            local_conf = None
            for key, conf in site_tier1.items():
                if conf['id'] == node_id:
                    local_conf = conf
                    break

            if local_conf is None:
                raise KeyError(f"ID {node_id} not found inside {site_name} tier1 list.")

            self.ip_address = local_conf['ip']
            self.device = local_conf['device']

        except KeyError:
            logger.error(f"Device {node_id} or Site {site_name} not found in topology.")
            raise Exception(f"Configuration Error: Device {node_id} not found in {site_name}")

        # 2. Find Manager (Tier 2 Server 1)
        try:
            tier2_conf = self.topology['sites'][site_name]['tier2']
            manager_key = next(iter(tier2_conf))  # gets first server (The Manager)
            manager_conf = tier2_conf[manager_key]

            self.manager_ip = manager_conf['ip']


            # Connect to the port reserved for devices (Index 1)
            # Eg: [9100, 9101] -> Uses 9101 for Device Communication
            if len(manager_conf['port']) < 2:
                raise Exception("Manager config missing Device Port (Index 1)!")

            self.manager_port_control = manager_conf['port'][1]
            self.manager_port_data = manager_conf['port'][2]


        except KeyError:
            raise Exception(f"Tier 2 Manager not found for {site_name}")

    def build_agent(self, data_dir, dataset_name):
        logger.info(f"Building Agent {self.node_id} targeting Manager at {self.manager_ip}:{self.manager_port_control}/{self.manager_port_data}")
        return NodeAgent(
            node_id=self.node_id,
            ip_address=self.ip_address,
            manager_ip=self.manager_ip,
            manager_port_control=self.manager_port_control,
            manager_port_data=self.manager_port_data,
            proxies=helpers.load_proxy_map(self.topology),
            data_dir = data_dir,
            dataset_name = dataset_name,
            device=self.device,
            new_size=self.new_size
        )