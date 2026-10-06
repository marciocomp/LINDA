# @author: Marcio Lopes
import argparse
import sys
import time
import signal


from numpy.random.mtrand import seed

sys.path.append('../../../../')
from config_node_agent import ConfigNodeAgent
from utils.linda_logger import logger
from utils.datasets.data_partitioner import DataPartitioner
from utils import helpers

def signal_handler(sig, frame):
    logger.info("Interrupt received. Exiting...")
    sys.exit(0)


def main():
    """
    Usage: python run_node_agent.py --site site3 --id s3_t1_1 --epochs 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, default="site1", help="Site Name")
    parser.add_argument("--id", type=str, default="s1_t1_1", help="Node ID")
    parser.add_argument("--data_dir", type=str, default="~/ALPACA", help="Path to store dataset")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--lab", type=str, default="local")
    parser.add_argument("--new_size", type=float, default=0.9)


    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("======================================")
    logger.info(f"STARTING NODE AGENT: {args.id} ({args.site})")
    logger.info("======================================")


    lab = args.lab
    if lab == "cds":
        topology_json_path = "../network_config_cds.json"
        if args.dataset == "imagenet":
            data_dir = "~/LINDA_data/imagenet-mini"
        else:
            data_dir = "~/LINDA_data"
    elif lab == "cds_res":
        topology_json_path = "../network_config_cds_resnet.json"
        if args.dataset == "imagenet":
            data_dir = "~/LINDA_data/imagenet-mini"
        else:
            data_dir = "~/LINDA_data"

    elif lab == "lrc":
        topology_json_path = "../network_config_lrc.json"
        # data_dir = "./ALPACA"
        if args.dataset =="imagenet":
            data_dir = "./LINDA_data/imagenet-mini"
        else:
            data_dir = "./LINDA_data"

    else:
        topology_json_path = "../network_config_local.json"
        if args.dataset =="imagenet":
            data_dir = "~/LINDA_data/imagenet-mini"
        else:
            data_dir = "~/LINDA_DATA"


    conf = ConfigNodeAgent(site_name=args.site,
                           node_id=args.id,
                           topology_json_path=topology_json_path,
                           new_size=args.new_size)

    agent = conf.build_agent(data_dir=data_dir,dataset_name=args.dataset)

    # 2. Connect (Retry Loop)
    connected = False
    i = 0
    attempts = 10
    while not connected and i < attempts:
        connected = agent.connect_to_manager_control()
        if not connected:
            i += 1
            logger.warning(f"Manager not ready. Retrying in 5s... ({i}/{attempts})")
            time.sleep(5)

    if not connected:
        logger.error("Failed to connect to Cluster Manager. Aborting.")
        sys.exit(1)

    # 3. Wait for Configuration & Prepare Data
    connected = False
    i = 0
    attempts = 10
    while not connected and i < attempts:
        connected = agent.connect_to_manager_data()
        if not connected:
            i += 1
            logger.warning(f"Manager not ready. Retrying in 5s... ({i}/{attempts})")
            time.sleep(2)

    if not connected:
        logger.error("Failed to connect to Cluster Manager. Aborting.")
        sys.exit(1)

    if agent.wait_for_instructions():

        logger.info("Agent configured successfully.")
        if agent.wait_for_start_signal():
            logger.info("Agent started successfully.")
            try:
                agent.train()
            except Exception as exc:

                helpers.report_fatal(
                    exc, node_id=agent.node_id, run_id=agent.run_id,
                    seed=agent.seed, client_id=agent.node_id, round_idx=-1,
                    batch_idx=-1, device=agent.device, phase="train",
                    layers="0-%s" % getattr(agent, "split_point", ""))


    logger.info("Stopping LINDA Tier 1...")


if __name__ == "__main__":
    main()