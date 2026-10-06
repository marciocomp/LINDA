# @author: Marcio Lopes

import argparse
import sys
import time

sys.path.append('../../../../')

from config_cluster_manager import ConfigClusterManager
from utils.linda_logger import logger


def main():
    """
    Entry point for Tier 2 Cluster Manager (Worker).
    Usage:
    python run_cluster_manager.py --site site2 --id s2_t2_1
    """

    parser = argparse.ArgumentParser(description="LINDA Tier 2 - Cluster Manager Runner")
    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site", type=str, default="site1")
    parser.add_argument("--id", type=str, default="s1_t2_1")
    parser.add_argument("--lab", type=str, default="local")


    args = parser.parse_args()
    # set_global_seed(args.seed)

    logger.info("=======================================")
    logger.info(f"   STARTING CLUSTER MANAGER: {args.site.upper()}   ")
    logger.info("=======================================")


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

    configurator = ConfigClusterManager(site_name=args.site,
                                        node_id=args.id,
                                        topology_json_path=topology_json_path)

    # 2. Build Component
    logger.info(f"Initializing Cluster Manager: {args.id} @ {args.site}")
    manager = configurator.build_manager()

    # 3. Connect to Leader (Tier 3) - CONTROL PLANE
    connected = False
    attempt = 0
    max_retries = 10

    while not connected and attempt < max_retries:
        connected = manager.connect_to_orchestrator_control()
        if not connected:
            attempt += 1
            logger.warning(f"Control Connection failed. Retrying in 5s... ({attempt}/{max_retries})")
            time.sleep(5)

    if not connected:
        logger.error("Failed to connect to Orchestrator Control Plane. Aborting.")
        sys.exit(1)

    # 3b. Accept the local Compute Nodes BEFORE asking for the allocation.
    n_workers = configurator.get_secondary_workers_count()
    manager.wait_for_local_workers(n_workers)

    # 3c. Resource discovery (Algorithm 1): report what this site measured.
    manager.report_site_profile()

    logger.info(f"Cluster Manager {args.site} connected. Waiting for Allocation...")

    # 4. Wait for Instructions (BLOCKING)
    if manager.wait_for_initial_allocation():
        logger.info("Allocation Received! Preparing Tier 2 Resources...")
    else:
        logger.error("Failed to receive configuration. Aborting.")
        sys.exit(1)
    # 5. Connect to Leader (Tier 3) - DATA PLANE
    connected = False
    attempt = 0
    # first = True
    while not connected and attempt < max_retries:
        connected = manager.connect_to_orchestrator_data()

        if not connected:
            attempt += 1
            logger.warning(f"Data Connection failed. Retrying in 5s... ({attempt}/{max_retries})")
            time.sleep(5)

    if not connected:
        logger.error("Failed to connect to Orchestrator Data Plane. Aborting.")
        sys.exit(1)

    # (the Compute Nodes were accepted at step 3b, before the allocation)
    manager.start_server_for_devices()

    manager.listen_worker_data()

    # time.sleep(1)

    logger.info("=== CLUSTER READY: Compute Nodes and Node Agents are now Connected ===")

    manager.distribute_intra_cluster_tasks()

    manager.wait_and_send_start_training()



        # manager.start_training_as_worker(iteration=estimated_batches * epochs)

    logger.info("Manager stopped.")


if __name__ == "__main__":
    main()