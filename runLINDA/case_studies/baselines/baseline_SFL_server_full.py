# @author: Marcio Lopes
# Same training path as baseline_SFL_server.py, plus the weight synchronization phase the
# published baseline does not have: the client shards are collected, averaged with
# the server-side shards (weighted FedAvg) and redistributed, so the cost of a
# federated round can be compared with LINDA's. Nothing in the training loop
# changed, so T_total_train_time stays comparable with the original baseline.
import torch
import torch.optim as optim
import tqdm
import time
import argparse
import os
import sys
from pathlib import Path
import gc
import datetime
import transformers

sys.path.append('../../../../..')
sys.path.append(str(Path(__file__).resolve().parents[5]))   # repo root, whatever the cwd


from utils.communication.communicationModule import CommunicationModule
from utils.linda_logger import logger
from utils.model_factory import ModelFactory
from utils.huggingface_token import HF_TOKEN
from utils import helpers
import weights_sync_full as wsync

RUN_ID = f'b_exp2_SFL_full_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline SFL Server with weight synchronization")
    parser.add_argument("--ip", type=str, default="130.92.70.8")
    parser.add_argument("--port", type=int, default=10101)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--model_type", type=str, default="llm")
    parser.add_argument("--split_point", type=int, default=1)
    parser.add_argument("--total_layers", type=int, default=18)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="alpaca")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--num_client_procs", type=int, default=2,
                        help="client processes to wait for; each announces the clients it owns, "
                             "so the four clients can be split across two hosts")

    return parser.parse_args()


def load_server_shard(args, device, client_idx: int):
    snapshot_start = helpers.get_local_resource_snapshot(device)
    logger.info(f"[Server] Snapshot BEFORE initialization (client_idx={client_idx}):\n\t\t\t{snapshot_start}")

    model = ModelFactory.get_model_shard(
        model_name=args.model_name,
        model_type=args.model_type,
        start_layer=args.split_point,
        end_layer=args.total_layers,
        is_first=False,
        is_last=True,
        device=device,
        hf_token=HF_TOKEN,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    snapshot_end = helpers.get_local_resource_snapshot(device)
    logger.info(f"[Server] Snapshot AFTER initialization (client_idx={client_idx}):\n\t\t\t{snapshot_end}")

    helpers.log_snapshots(
        node_id="server_0",
        run_id=RUN_ID,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        description="Model_Shard_Load",
        r=client_idx,   # <-- usa client_idx para não ficar tudo em -1
        i=-1
    )

    return model, optimizer, criterion


def train_step(server, client_sock, device, model, optimizer, criterion, iterations: int, seed: int, client_id: str, client_idx: int, clip_max_norm: float = 1.0):
    model.train()
    snapshot_start = helpers.get_local_resource_snapshot(device)

    for i in tqdm.tqdm(range(iterations), desc=f"[server_0 | {client_id}]"):
        msg = server.recv_msg(client_sock, "FORWARD_DATA")
        payload = msg[1]

        input_tensor = payload["data"].to(device)
        labels = payload["labels"].to(device)

        mask = payload.get("attention_mask")
        mask = mask.to(device) if mask is not None else None

        pos_ids = payload.get("position_ids")
        pos_ids = pos_ids.to(device) if pos_ids is not None else None

        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)

        t_start_comp = time.perf_counter()

        outputs = model(input_tensor, attention_mask=mask, position_ids=pos_ids)

        logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step()

        helpers.sync_device(device)

        t_end_comp = time.perf_counter()
        helpers.log_performance_metric(
            node_id="server_0",
            client_id=client_id,
            seed=seed,
            run_id=RUN_ID,
            metric_name="T_comp",
            duration=(t_end_comp - t_start_comp),
            round_idx=client_idx,
            batch_idx=i,
            req_id=None,
            delay=None
        )

        grad_cpu = input_tensor.grad.detach().cpu()
        input_tensor.grad = None

        response_payload = {"grad": grad_cpu}

        t0 = time.perf_counter()
        server.send_msg(client_sock, ["BACKWARD_DATA", response_payload])
        t1 = time.perf_counter()

        helpers.log_performance_metric(
            node_id="server_0",
            client_id=client_id,
            seed=seed,
            run_id=RUN_ID,
            metric_name="T_comm",
            duration=(t1 - t0),
            round_idx=client_idx,
            batch_idx=i,
            req_id=None,
            delay=None
        )

        snapshot_end = helpers.get_local_resource_snapshot(device)
        helpers.log_snapshots(
            node_id="server_0",
            run_id=RUN_ID,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            r=client_idx,
            i=i,
            description="After_backward"
        )

        # limpeza leve por batch
        del outputs, logits, shift_labels, loss, grad_cpu, response_payload
        del input_tensor, labels


def do_connection(ip, port, num_client_procs, num_clients):
    """
    Waits for the client processes and learns which clients each one owns, so the
    four clients can be split across two hosts and face the same host contention
    as the other architectures. A process that announces nothing is assumed to own
    every client, which keeps the single-process usage working.
    """
    logger.info(f"[Server] Waiting for {num_client_procs} client process(es) on {ip}:{port}")
    server = CommunicationModule("server_0", ip)
    server.enable_net_metrics(RUN_ID)
    server.sock_server.bind((ip, port))
    server.sock_server.listen(max(1, num_client_procs))

    socks, sock_by_client = [], {}
    for _ in range(num_client_procs):
        client_sock, addr = server.sock_server.accept()
        msg = server.recv_msg(client_sock, "HELLO_SERVER")
        payload = msg[1] if msg and len(msg) > 1 and isinstance(msg[1], dict) else {}
        owned = [int(i) for i in payload.get("client_ids", [])] or list(range(num_clients))
        socks.append(client_sock)
        for idx in owned:
            sock_by_client[idx] = client_sock
        logger.info(f"[Server] Handshake from {addr}: clients={owned}")

    missing = [i for i in range(num_clients) if i not in sock_by_client]
    if missing:
        raise RuntimeError(f"[Server] no client process announced clients {missing}")

    return server, socks, sock_by_client


def _cleanup_model(device, model=None, optimizer=None, criterion=None):
    try:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass
    try:
        del model, optimizer, criterion
    except Exception:
        pass

    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_id = int(args.device.split(":")[1])
        torch.cuda.set_device(gpu_id)
    initial_snapshot = helpers.get_local_resource_snapshot(device)

    server, client_socks, sock_by_client = do_connection(
        args.ip, args.port, args.num_client_procs, args.num_clients)

    server_packs = []
    samples_by_client = {}

    for client_idx in range(args.num_clients):
        client_sock = sock_by_client[client_idx]

        cfg_msg = server.recv_msg(client_sock, "CLIENT_CONFIG")
        cfg = cfg_msg[1]
        client_id = cfg["client_id"]
        iterations = int(cfg["iterations"])
        logger.info(f"[Server] Starting client loop {client_idx}/{args.num_clients-1}: {client_id}, iterations={iterations}")


        model, optimizer, criterion = load_server_shard(args, device, client_idx=client_idx)
        model.train()


        server.send_msg(client_sock, ["START_CLIENT", {"run_id": RUN_ID, "client_id": client_id}])

        start_training = time.perf_counter()

        train_step(
            server=server,
            client_sock=client_sock,
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            iterations=iterations,
            seed=args.seed,
            client_id=client_id,
            client_idx=client_idx,
            clip_max_norm=args.clip_max_norm
        )

        end_training = time.perf_counter()
        total_training_time = end_training - start_training

        helpers.log_performance_metric(
            node_id="server_0",
            client_id=client_id,
            seed=args.seed,
            run_id=RUN_ID,
            metric_name="T_total_train_time",
            duration=total_training_time,
            round_idx=client_idx,
            batch_idx=iterations,
            req_id=None,
            delay=None
        )

        logger.info(f"[Server] Client {client_id} finished. Total time={total_training_time:.2f}s")

        samples_by_client[client_id] = iterations * args.batch_size
        server_packs.append(wsync.pack(model, client_id, "t2", args.split_point, args.total_layers))

        _cleanup_model(device, model=model, optimizer=optimizer, criterion=criterion)


        server.send_msg(client_sock, ["CLIENT_DONE_ACK", {"client_id": client_id}])


    # ------------------------------------------------------------------
    # Weight synchronization (absent from the published baseline)
    # ------------------------------------------------------------------
    logger.info("[Server] === Weight synchronization ===")
    timer = wsync.PhaseTimer(helpers, "server_0", RUN_ID, args.seed)

    for sock in client_socks:
        server.send_msg(sock, [wsync.REQUEST, {"round": 0, "clients": list(samples_by_client)}])

    client_packs = []
    for client_idx in range(args.num_clients):
        msg = server.recv_msg(sock_by_client[client_idx], wsync.RESPONSE)
        if not msg:
            logger.error("[Server] client weights missing; synchronization aborted.")
            break
        client_packs.append(msg[1])
    timer.phase("T_collect_weight")

    averaged = wsync.fed_avg(client_packs + server_packs, samples_by_client)
    if not averaged:
        logger.warning("[Server] nothing to average: every client reported zero samples, "
                       "which happens when the dataset is too small for one batch.")
    logger.info(f"[Server] averaged {len(averaged)} shard(s) over {len(samples_by_client)} clients.")
    timer.phase("T_aggregation")

    client_avg = wsync.match(averaged, "t1", 0, args.split_point)
    for client_idx in range(args.num_clients):
        server.send_msg(sock_by_client[client_idx], [wsync.UPDATE, client_avg])
    for sock in client_socks:
        server.send_msg(sock, [wsync.DONE, {"reason": "global_update_applied"}])
    timer.phase("T_distribution_weight")
    timer.total()

    for sock in client_socks:
        server.send_msg(sock, ["ALL_DONE", {"run_id": RUN_ID}])
        sock.close()

    metadata = {
        "meta": {
            "run_id": RUN_ID,
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": args.seed,
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(device),
            "initial_system_snapshot": initial_snapshot
        },
        "training_phase": {
            "rounds": args.rounds,
            "num_clients": args.num_clients
        },
        "model": {
            "name": args.model_name,
            "type": args.model_type,
            "total_layers": args.total_layers,
            "batch_size": args.batch_size,
        },
    }

    helpers.save_experiment_metadata("server_0", RUN_ID, metadata)
    logger.info("[Server] === Session Finished ===")


if __name__ == "__main__":
    main()