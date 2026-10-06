# @author: Marcio Lopes (patched for sequential clients)
import torch
import tqdm
import torch.optim as optim
import time
import argparse
import sys
import datetime
import os
import gc
import transformers

sys.path.append('../../../../')
from torch.utils.data import DataLoader
from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils.datasets.alpaca import AlpacaDataset, collate_fn
from utils.datasets.data_partitioner import DataPartitioner
from utils import helpers


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline SFL Client (Sequential Clients)")
    parser.add_argument("--ip_server", type=str, default="130.92.70.8")
    parser.add_argument("--ip", type=str, default="130.92.65.237")
    parser.add_argument("--port_server", type=int, default=10101)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--model_type", type=str, default="llm")
    parser.add_argument("--split_point", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--dataset", type=str, default="alpaca")
    parser.add_argument("--data_dir", type=str, default="~/LINDA_data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--new_size", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--partition_method", type=str, default="content_dirichlet_balanced")

    return parser.parse_args()


def load_client_shard(args, device):
    logger.info(f"Loading Client Shard: Layers 0 to {args.split_point}")
    model = ModelFactory.get_model_shard(
        model_name=args.model_name,
        model_type=args.model_type,
        start_layer=0,
        end_layer=args.split_point,
        is_first=True,
        is_last=False,
        device=device,
        hf_token=HF_TOKEN,
    )
    return model


def _prepare_data_for_client(args, client_idx: int):
    logger.info(f"[Client {client_idx}] Preparing Dataset...")

    dataset_name = "tatsu-lab/alpaca"
    collate = collate_fn

    eval_indices, train_pool_indices, universe_indices = DataPartitioner.get_global_eval_and_train_pool(
        dataset_name=dataset_name,
        new_size=args.new_size,
        eval_fraction=0.10,
        seed=args.seed,
        cache_dir=args.data_dir
    )

    train_indices = DataPartitioner.get_partition_indices(
        dataset_name=dataset_name,
        total_clients=args.num_clients,
        partition_id=client_idx,
        method=args.partition_method,
        alpha=args.alpha,
        seed=args.seed,
        cache_dir=args.data_dir,
        pool_indices=train_pool_indices
    )

    train_dataset = AlpacaDataset(
        tokenizer_name=args.model_name,
        split="train",
        indices=train_indices,
        cache_dir=args.data_dir
    )

    val_dataset = AlpacaDataset(
        tokenizer_name=args.model_name,
        split="train",
        indices=eval_indices,
        cache_dir=args.data_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True
    )

    iterations = len(train_loader)

    logger.info(
        f"[Client {client_idx}] Ready: TrainSamples={len(train_dataset)} | "
        f"TrainBatches={len(train_loader)} | EvalSamples={len(val_dataset)} | Iterations={iterations}"
    )

    return train_dataset, val_dataset, train_loader, iterations


def do_connection(ip, port, ip_server, port_server):
    logger.info(f"[Client] Connecting to {ip_server}:{port_server}")
    client = CommunicationModule("client_runner", ip)
    client.sock_client.connect((ip_server, port_server))
    server_sock = client.sock_client
    client.send_msg(server_sock, ["HELLO_SERVER", {"ok": True}])
    logger.info("[Client] Connected to server!")
    return client, server_sock


def train_step(device, model, optimizer, client, server_sock, train_loader, seed, run_id, client_id, client_idx, clip_max_norm=1.0):
    logger.info(f"[{client_id}] === Starting Split Learning Loop ===")
    model.train()
    snapshot_start = helpers.get_local_resource_snapshot(device)

    for i, batch in enumerate(tqdm.tqdm(train_loader, desc=f"[{client_id}]")):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        output = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        helpers.sync_device(device)

        t1 = time.perf_counter()

        helpers.log_performance_metric(
            node_id=client_id,
            client_id=client_id,
            seed=seed,
            run_id=run_id,
            metric_name="T_comp",
            duration=(t1 - t0),
            round_idx=client_idx,
            batch_idx=i,
            req_id=None,
            delay=None
        )

        payload = {
            "data": output.detach().cpu(),
            "labels": labels.cpu(),
            "attention_mask": attention_mask.cpu(),
            "position_ids": position_ids.cpu(),
        }

        t0c = time.perf_counter()
        client.send_msg(server_sock, ["FORWARD_DATA", payload])
        t1c = time.perf_counter()

        helpers.log_performance_metric(
            node_id=client_id,
            client_id=client_id,
            seed=seed,
            run_id=run_id,
            metric_name="T_comm",
            duration=(t1c - t0c),
            round_idx=client_idx,
            batch_idx=i,
            req_id=None,
            delay=None
        )

        # snapshots opcionais (mantive)
        snapshot_end = helpers.get_local_resource_snapshot(device)
        helpers.log_snapshots(
            node_id=client_id,
            run_id=run_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            r=client_idx,
            i=i,
            description="After_forward"
        )

        msg = client.recv_msg(server_sock, "BACKWARD_DATA")
        grad_tensor = msg[1]["grad"].to(device).to(dtype=output.dtype)

        tb0 = time.perf_counter()
        output.backward(grad_tensor)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step()
        helpers.sync_device(device)

        tb1 = time.perf_counter()

        helpers.log_performance_metric(
            node_id=client_id,
            client_id=client_id,
            seed=seed,
            run_id=run_id,
            metric_name="T_comp_backward",
            duration=(tb1 - tb0),
            round_idx=client_idx,
            batch_idx=i,
            req_id=None,
            delay=None
        )

        snapshot_end = helpers.get_local_resource_snapshot(device)
        helpers.log_snapshots(
            node_id=client_id,
            run_id=run_id,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            r=client_idx,
            i=i,
            description="After_backward"
        )

        # limpeza leve por batch
        del output, grad_tensor, payload, msg, input_ids, attention_mask, labels, position_ids

    logger.info(f"[{client_id}] Finished.")


def _cleanup_client(device, model=None, optimizer=None):
    try:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass
    try:
        del model, optimizer
    except Exception:
        pass
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    device = torch.device(args.device)

    client, server_sock = do_connection(
        ip=args.ip,
        port=args.port,
        ip_server=args.ip_server,
        port_server=args.port_server
    )

    for client_idx in range(args.num_clients):
        client_id = f"client_{client_idx}"

        train_dataset, val_dataset, train_loader, iterations = _prepare_data_for_client(args, client_idx)

        client.send_msg(server_sock, ["CLIENT_CONFIG", {"client_id": client_id, "iterations": iterations}])

        msg = client.recv_msg(server_sock, "START_CLIENT")
        run_id = msg[1]["run_id"]
        client.enable_net_metrics(run_id)
        logger.info(f"[{client_id}] START_CLIENT received. run_id={run_id}")

        model = load_client_shard(args, device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        t0 = time.perf_counter()
        train_step(
            device=device,
            model=model,
            optimizer=optimizer,
            client=client,
            server_sock=server_sock,
            train_loader=train_loader,
            seed=args.seed,
            run_id=run_id,
            client_id=client_id,
            client_idx=client_idx,
            clip_max_norm=args.clip_max_norm
        )
        t1 = time.perf_counter()

        total_training_time = t1 - t0
        helpers.log_performance_metric(
            node_id=client_id,
            client_id=client_id,
            seed=args.seed,
            run_id=run_id,
            metric_name="T_total_train_time",
            duration=total_training_time,
            round_idx=client_idx,
            batch_idx=iterations,
            req_id=None,
            delay=None
        )

        _cleanup_client(device, model=model, optimizer=optimizer)

        _ = client.recv_msg(server_sock, "CLIENT_DONE_ACK")

        metadata = {
            "meta": {
                "run_id": run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "seed": args.seed,
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device": helpers.get_local_device_name(device),
                "new_size": args.new_size,
                "alpha": args.alpha,
                "partition_method": args.partition_method,
                "client_idx": client_idx,
                "client_id": client_id,
            },
            "dataset details": {
                "Name": args.dataset,
                "Train Samples": len(train_dataset),
                "Train Batches": len(train_loader),
                "Eval Samples": len(val_dataset),
                "Iterations": iterations,
            }
        }
        helpers.save_experiment_metadata(node_id=client_id, run_id=run_id, metadata=metadata)

        logger.info(f"[{client_id}] Done. Total time={total_training_time:.2f}s")

    msg = client.recv_msg(server_sock, "ALL_DONE")
    logger.info(f"[Client] ALL_DONE received: {msg[1]}")
    server_sock.close()


if __name__ == "__main__":
    main()