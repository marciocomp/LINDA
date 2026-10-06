# @author: Marcio Lopes
import pickle
import struct
import socket
import time

from utils.linda_logger import logger
from utils.communication.net_metrics import NetRecorder, read_rx_queued, read_tcp_info


class CommunicationModule(object):
    """
    Base Communication Class.
    Handles low-level socket operations with object serialization (Pickle).
    """

    def __init__(self, node_id, ip_address):
        self.node_id = node_id
        self.ip = ip_address
        self.time_delta = 0.0

        self.sock_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.sock_server_data = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_server_data.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.orchestrator_data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.orchestrator_data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.sock_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.sock_manager_to_device = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_manager_to_device.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.sock_neighbor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_neighbor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Per-message network metrics (utils/communication/net_metrics.py)
        self._net = NetRecorder(node_id)

    def enable_net_metrics(self, run_id):
        """Starts writing the network records under results/{run_id}/ once the run id is known."""
        net = getattr(self, "_net", None)
        if net is not None:
            net.enable(run_id)

    def recv_msg(self, sock, expect_msg_type=None):
        """
        Receives a message: [Header: 4 bytes size] + [Payload: Pickled Object]
        """
        # Instances built without __init__ (e.g. the calibration selftest) record nothing
        net = getattr(self, "_net", None)
        try:
            ts_start = time.time()
            t0 = time.perf_counter()

            raw_msglen = self._recvall(sock, 8)
            if not raw_msglen:
                return None
            msg_len = struct.unpack(">Q", raw_msglen)[0]

            # Header in hand: everything before was idle wait, everything after is the body transfer
            t1 = time.perf_counter()
            rx_queued = read_rx_queued(sock) if net is not None else None
            info_start = read_tcp_info(sock) if net is not None else None

            msg_data = self._recvall(sock, msg_len)

            t2 = time.perf_counter()
            info_end = read_tcp_info(sock) if net is not None else None

            msg = pickle.loads(msg_data)

            if net is not None:
                net.record_recv(sock, msg, 8 + msg_len, ts_start, self.time_delta,
                                t1 - t0, t2 - t1, time.perf_counter() - t2,
                                rx_queued, info_start, info_end)

            if expect_msg_type is not None:
                if isinstance(msg, (list, tuple)) and len(msg) > 0:
                    if msg[0] != expect_msg_type:
                        logger.warning(f"Expected {expect_msg_type} but got {msg[0]}")

            return msg

        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    def _recvall(self, sock, n):
        """Helper to ensure we read exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def send_msg(self, sock, msg):
        """
        Sends a message: [Header: 4 bytes size] + [Payload: Pickled Object]
        """
        net = getattr(self, "_net", None)
        try:
            ts_start = time.time()
            t0 = time.perf_counter()

            msg_pickle = pickle.dumps(msg)
            # 'Q' = unsigned long long (8 bytes) -> Max 18 Exabytes
            # '!Q' = Network (Big-Endian) unsigned long long
            frame = struct.pack(">Q", len(msg_pickle)) + msg_pickle

            t1 = time.perf_counter()
            info_start = read_tcp_info(sock) if net is not None else None

            sock.sendall(frame)

            t2 = time.perf_counter()
            if net is not None:
                net.record_send(sock, msg, len(frame), ts_start, self.time_delta,
                                t1 - t0, t2 - t1, info_start, read_tcp_info(sock))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def perform_time_sync_handshake(self, sock, role="client"):
        """
        Clock synchronization using simplified Cristian's algorithm.
        Role='server': Orchestrator or Manager (time reference)
        Role='client': Manager or Worker (adjusts its clock)
        """
        try:
            if role == "server":
                msg = self.recv_msg(sock, "TIME_SYNC_PING")
                t2 = time.time()
                # t2 = self.get_synced_time()

                if not msg: return

                t1 = msg[1]['t1']

                t3 = time.time()
                # t3 = self.get_synced_time()
                payload = {'t1': t1, 't2': t2, 't3': t3}
                self.send_msg(sock, ["TIME_SYNC_PONG", payload])
                logger.info(f"[TimeSync] Server responded to sync request from socket.")

            elif role == "client":
                t1 = time.time()
                self.send_msg(sock, ["TIME_SYNC_PING", {'t1': t1}])

                msg = self.recv_msg(sock, "TIME_SYNC_PONG")
                t4 = time.time()

                if msg:
                    data = msg[1]
                    t1_remote, t2, t3 = data['t1'], data['t2'], data['t3']

                    # RTT = (t4 - t1) - (t3 - t2)
                    # Offset = ((t2 - t1) + (t3 - t4)) / 2

                    offset = ((t2 - t1) + (t3 - t4)) / 2
                    self.time_delta = offset
                    rtt = (t4 - t1) - (t3 - t2)
                    net = getattr(self, "_net", None)
                    if net is not None:
                        net.record_timesync(sock, t1, rtt, offset)
                    logger.info(f"[TimeSync] Synchronized! Local correction delta: {self.time_delta:.6f}s "
                                f"| RTT: {rtt * 1e3:.3f} ms")
        except Exception as e:
            logger.error(f"[TimeSync] Failed: {e}")

    def get_synced_time(self):
        """Returns the timestamp adjusted to the master clock."""
        return time.time() + self.time_delta