# @author: Marcio Lopes
import argparse
import sys
import time

sys.path.append('../../../../')

from config_compute_node import ConfigComputeNode
from utils.linda_logger import logger


def main():
    parser = argparse.ArgumentParser(description="LINDA Tier 2 - Compute Node Runner")
    parser.add_argument("--site", type=str, default="site1", help="Site ID")
    parser.add_argument("--id", type=str, required=True, help="Server ID (e.g., s1_t2_2)")
    parser.add_argument("--lab", type=str, default="local")

    args = parser.parse_args()

    logger.info("=====================================")
    logger.info(f"   STARTING COMPUTE NODE: {args.id}   ")
    logger.info("=====================================")

    # 1. Configure
    lab = args.lab
    if lab == "cds":
        topology_json_path = "../network_config_cds.json"
    elif lab == "cds_res":
        topology_json_path = "../network_config_cds_resnet.json"
    elif lab == "lrc":
        topology_json_path = "../network_config_lrc.json"
    else:
        topology_json_path = "../network_config_local.json"

    configurator = ConfigComputeNode(site_name=args.site,
                                         server_id=args.id,
                                         topology_json_path=topology_json_path)

    # 2. Build
    node = configurator.build_node()

    # 3. Connect to Manager
    connected = False
    attempts = 0
    while not connected and attempts < 10:
        connected = node.connect_to_manager()
        if not connected:
            attempts += 1
            logger.warning(f"Connection to Manager failed. Retrying... ({attempts}/10)")
            time.sleep(5)

    if not connected:
        logger.error("Failed to connect to Cluster Manager. Aborting.")
        sys.exit(1)

    node.connect_manager_data()

    # 4. Wait for Configuration (Model Shard info & Next Hop)
    if node.wait_for_configuration():

        logger.info("Compute Node Configured and Ready.")
        node.listen_for_previous_hop()

        if node.wait_start_training():

            node.start_training_as_worker()

    else:
        logger.error("Failed to receive valid configuration.")

    logger.info("========= COMPUTE NODE FINISHED =========")



if __name__ == "__main__":
    main()