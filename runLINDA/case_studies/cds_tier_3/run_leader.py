# @author: Marcio Lopes

import argparse
import sys
import random
import numpy as np
import torch
import socket

sys.path.append('../../../../')

from config_leader import ConfigLeader
from utils.linda_logger import logger


def set_global_seed(seed):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    if seed is None:
        return

    logger.info(f"Setting Global Seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """
    Entry point for Tier 3 Global Orchestrator (Leader).

    Usage Examples:
    1. Vision: python run_leader.py --model_name resnet50 --model_type vision --rounds 50
    2. LLM:    python run_leader.py --model_name google/gemma-2b --model_type llm --rounds 10
    """

    parser = argparse.ArgumentParser(description="LINDA Tier 3 - Global Leader Runner")

    # --- Infrastructure Arguments ---
    parser.add_argument("--site", type=str, default="site1")
    parser.add_argument("--straggler", type=str, default=True)

    # --- Experiment Control ---
    parser.add_argument("--seed", type=int, default=20260311)

    # Model Configuration
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")#"resnet50") #"google/gemma-7b")
    parser.add_argument("--model_type", type=str, default="llm", choices=['vision', 'llm'], help="Type of model architecture")
    parser.add_argument("--dataset", type=str, default= "alpaca")

    # --- Hyperparameters ---
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer (default: 0.01)")
    parser.add_argument("--mu", type=float, default=0,
                        help="FedProx proximal term (mu=0 means pure FedAvg; mu=0.01 is FedProx)")
    parser.add_argument("--fedprox_cpu", action="store_true",
                        help="Keep FedProx global params on CPU to save VRAM")
    parser.add_argument("--lab", type=str, default="cds")
    parser.add_argument("--realloc", action="store_true",
                        help="Re-plan the allocation at a round boundary when the "
                             "measured resources call for it. Off by default, which "
                             "is the behaviour that keeps the initial state.")
    parser.add_argument("--realloc_min_rounds", type=int, default=1,
                        help="How many consecutive rounds must call for the same "
                             "change before it is acted on.")


    args = parser.parse_args()

    # 1. Set Reproducibility
    set_global_seed(args.seed)

    # 2. Package Hyperparameters
    hyperparams = {
        "seed": args.seed,
        "rounds": args.rounds,
        "site": args.site,
        # "straggler": args.straggler,
        # "local_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "alpha": args.alpha,
        "model_name": args.model_name,
        "model_type": args.model_type,
        # "dataset": args.dataset,
        "lab": args.lab,
        "num_classes": 1000 if args.dataset == "imagenet" else 10,
        "weight_decay": args.weight_decay,
        "mu": args.mu,
        "fedprox_cpu": args.fedprox_cpu,
        "realloc": args.realloc,
        "realloc_min_rounds": args.realloc_min_rounds,
    }

    # 3. Instantiate Configuration
    lab = args.lab
    if lab == "cds":
        topology_json_path = "../network_config_cds.json"
    elif lab == "cds_res":
        topology_json_path = "../network_config_cds_resnet.json"
    elif lab == "lrc":
        topology_json_path = "../network_config_lrc.json"
    else:
        topology_json_path = "../network_config_local.json"


    configurator = ConfigLeader(hyperparams=hyperparams,
                                topology_json_path=topology_json_path)


    # 4. Start Execution
    logger.info(f"Initializing LINDA Global Orchestrator [{args.model_type.upper()}: {args.model_name}]")
    logger.info("Awaiting Resource Discovery to determine Split Matrix...")


    orchestrator = configurator.build_orchestrator()

    expected_clusters = len(orchestrator.topology['sites'])
    orchestrator.do_cluster_connections(expected_clusters)

    is_train_possible = configurator.discover_and_allocate(orchestrator)

    if is_train_possible:
        logger.info("==============================================")
        logger.info("======= SYSTEM INITIALIZATION COMPLETE =======")
        logger.info("==============================================")

        if orchestrator.distribute_initial_context():
            orchestrator.start_global_training_loop()
    else:
        logger.error("============================================")
        logger.error("======= SYSTEM INITIALIZATION FAILED =======")
        logger.error("============================================")

    logger.info("Stopping LINDA Tier 3...")

if __name__ == "__main__":
    logger.info("=======================================")
    logger.info("======= STARTING LEADER CLUSTER =======")
    logger.info("=======================================")
    main()