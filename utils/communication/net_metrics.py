# @author: Marcio Lopes
"""
Network metrics recorded inside CommunicationModule.send_msg and recv_msg, so
that payload size, throughput and RTT come from the run itself and not from a
separate probe (runLINDA/case_studies/cds/baselines/00_review/network_calibration.py
is only the reference the run is checked against).

Hot path, per message: two TCP_INFO reads (plus one FIONREAD on receive), the
socket addresses and one queue put. The message object itself is never kept.
Link lookups, parsing and CSV writes run on a background thread that starts
once the run id is known; records produced before that (setup messages) wait
in the queue.

Output, under the same root as every other writer (utils/helpers.results_path):
  ../../results/{run_id}/network_metrics/{node_id}_{hostname}_{run_id}_network.csv   one row per message
  ../../results/{run_id}/network_metrics/{node_id}_{hostname}_{run_id}_links.csv     one row per TCP connection
  ../../results/{run_id}/network_metrics/{node_id}_{hostname}_{run_id}_timesync.csv  one row per clock sync

The hostname is part of the file name because baseline processes on different
hosts can share a node id (e.g. "client_runner").
"""
import atexit
import csv
import os
import platform
import queue
import socket
import struct
import subprocess
import threading

from utils.linda_logger import logger
from utils import helpers

try:
    import fcntl
    import termios
except ImportError:  # not Linux: no receive-queue reading
    fcntl = None
    termios = None

HEADER_BYTES = 8
FLUSH_INTERVAL_S = 1.0
CLOSE_TIMEOUT_S = 30.0
# Records kept while the run id is still unknown. Setup needs a few dozen; the
# cap only matters for a process that never enables recording.
MAX_PENDING = 5000

# ---------------------------------------------------------------------------
# Message classification
# ---------------------------------------------------------------------------
# Data messages carry tensors; data_kind separates them for the analysis.
DATA_MESSAGES = {
    # activations of the training forward pass
    "FORWARD_DATA": "activation",
    "FORWARD_DATA_NEXT_HOP": "activation",
    "FORWARD_DATA_TO_TIER_3": "activation",
    # gradients of the training backward pass
    "BACKWARD_DATA": "gradient",
    "BACKWARD_DATA_FROM_ORCH": "gradient",
    "BACKWARD_DATA_FROM_CLUSTER_MANAGER": "gradient",
    "BACKWARD_DATA_FROM_PREVIOUS_NODE": "gradient",
    # model weights, collected and redistributed once per round
    "LOCAL_WEIGHTS_RESPONSE": "weights",
    "LOCAL_WEIGHTS_RESPONSE_TAGGED": "weights",
    "SITE_WEIGHTS_RESPONSE": "weights",
    "UPDATE_GLOBAL_WEIGHTS": "weights",
    "UPDATE_LOCAL_WEIGHTS": "weights",
    "UPDATE_LOCAL_WEIGHTS_TAGGED": "weights",
    # activations of the evaluation pass
    "FORWARD_EVAL": "eval",
    "FORWARD_EVAL_NEXT_HOP": "eval",
    "FORWARD_EVAL_TO_TIER_3": "eval",
    # A re-planned allocation travels with the parameters of every new range, so it
    # is weight traffic, not a control message.
    "UPDATE_ALLOCATION": "weights",
    "RECONFIGURE_NODE": "weights",
}

# Everything else the LINDA tiers and the SFL/HSFL baselines exchange. Requests
# for weights are control (small); the responses carrying them are data.
CONTROL_MESSAGES = frozenset({
    # connection setup
    "HELLO_DEVICE", "HELLO_DEVICE_DATA", "HELLO_DATA", "HELLO_DATA_CLIENT",
    "HELLO_TIER2", "HELLO_COMPUTE", "HELLO_CLIENT_TRAIN", "HELLO_CLIENT_SLOT",
    "HELLO_DATA_SLOT", "HELLO_SERVER",
    "TIME_SYNC_PING", "TIME_SYNC_PONG", "PREVIOUS_HOP",
    "CONNECT_MANAGER_FORWARD_DATA", "RUN_ID",
    # configuration
    # resource discovery: the readings a site takes on its own devices, and the
    # request that asks for them. Small dictionaries, so control rather than data.
    "REQUEST_SITE_PROFILE", "SITE_PROFILE_REPORT",
    "REQUEST_NODE_PROFILE", "NODE_PROFILE_REPORT",
    # runtime re-allocation: KEEP_* say that nothing changes, which still has to be
    # said, and the two DONE messages confirm a new placement was applied.
    "KEEP_ALLOCATION", "KEEP_NODE_CONFIG",
    "UPDATE_ALLOCATION_DONE", "RECONFIGURE_NODE_DONE",
    "INITIAL_ALLOCATION", "INITIAL_ALLOCATION_COMPLETE", "AGENT_CONFIG",
    "COMPUTE_CONFIG", "CLIENT_LIST", "ITERATIONS", "CLIENT_CONFIG", "CYCLE_CONFIG",
    # sequencing
    "START_TRAINING", "ROUND_FINISH", "WORKER_TRAIN_DONE", "START_CYCLE",
    "CYCLE_READY", "CYCLE_DONE", "START_CLIENT", "CLIENT_DONE_ACK", "ALL_DONE",
    "UPDATE_DONE",
    # requests and evaluation bookkeeping
    "REQUEST_LOCAL_WEIGHTS", "REQUEST_LOCAL_WEIGHTS_TAGGED", "REQUEST_SITE_WEIGHTS",
    "EVAL_CLIENTS", "START_EVALUATION", "SKIP_EVALUATION", "EVAL_METRICS",
})


def classify(msg_type):
    """Returns (msg_class, data_kind) for a message name."""
    if msg_type in DATA_MESSAGES:
        return "data", DATA_MESSAGES[msg_type]
    if msg_type in CONTROL_MESSAGES:
        return "control", ""
    return "unclassified", ""


def describe(msg):
    """Extracts (msg_type, req_id, client_id, round) without keeping the message."""
    payload = None
    if isinstance(msg, (list, tuple)) and msg:
        head = msg[0]
        msg_type = head if isinstance(head, str) else type(head).__name__
        payload = msg[1] if len(msg) > 1 else None
    elif isinstance(msg, str):
        msg_type = msg  # e.g. "INITIAL_ALLOCATION_COMPLETE" is sent bare
    else:
        msg_type = type(msg).__name__

    req_id = client_id = round_idx = ""
    if isinstance(payload, dict):
        req_id = _scalar(payload.get("req_id"))
        client_id = _scalar(payload.get("original_source")
                            or payload.get("source")
                            or payload.get("client_id"))
        round_idx = _scalar(payload.get("round"))
    return msg_type, req_id, client_id, round_idx


def _scalar(value):
    return value if isinstance(value, (str, int)) and not isinstance(value, bool) else ""


# ---------------------------------------------------------------------------
# Kernel readings (no privileges needed)
# ---------------------------------------------------------------------------
# struct tcp_info from include/uapi/linux/tcp.h, in order and without padding.
# Fields were appended over kernel versions, so only those inside the length
# the kernel returns are read.
_TCP_INFO_LAYOUT = (
    ("state", "B"), ("ca_state", "B"), ("retransmits", "B"), ("probes", "B"),
    ("backoff", "B"), ("options", "B"), ("wscale", "B"), ("flags", "B"),
    ("rto", "I"), ("ato", "I"), ("snd_mss", "I"), ("rcv_mss", "I"),
    ("unacked", "I"), ("sacked", "I"), ("lost", "I"), ("retrans", "I"), ("fackets", "I"),
    ("last_data_sent", "I"), ("last_ack_sent", "I"), ("last_data_recv", "I"), ("last_ack_recv", "I"),
    ("pmtu", "I"), ("rcv_ssthresh", "I"), ("rtt", "I"), ("rttvar", "I"),
    ("snd_ssthresh", "I"), ("snd_cwnd", "I"), ("advmss", "I"), ("reordering", "I"),
    ("rcv_rtt", "I"), ("rcv_space", "I"), ("total_retrans", "I"),
    ("pacing_rate", "Q"), ("max_pacing_rate", "Q"), ("bytes_acked", "Q"), ("bytes_received", "Q"),
    ("segs_out", "I"), ("segs_in", "I"), ("notsent_bytes", "I"), ("min_rtt", "I"),
    ("data_segs_in", "I"), ("data_segs_out", "I"), ("delivery_rate", "Q"),
    ("busy_time", "Q"), ("rwnd_limited", "Q"), ("sndbuf_limited", "Q"),
)


def _field_offsets(layout):
    table, offset = {}, 0
    for name, fmt in layout:
        size = struct.calcsize("=" + fmt)
        table[name] = (offset, "=" + fmt, size)
        offset += size
    return table


_TCP_INFO_FIELDS = _field_offsets(_TCP_INFO_LAYOUT)
_TCP_INFO_BUFLEN = 512
_TCP_INFO_OPT = getattr(socket, "TCP_INFO", None)


def read_tcp_info(sock):
    """Raw struct tcp_info bytes, or None when unavailable (not TCP, not Linux)."""
    if _TCP_INFO_OPT is None:
        return None
    try:
        return sock.getsockopt(socket.IPPROTO_TCP, _TCP_INFO_OPT, _TCP_INFO_BUFLEN)
    except (OSError, AttributeError, TypeError):
        return None


def read_rx_queued(sock):
    """Bytes already waiting in the receive queue, or None when unavailable."""
    if fcntl is None:
        return None
    try:
        raw = fcntl.ioctl(sock.fileno(), termios.FIONREAD, b"\0\0\0\0")
        return struct.unpack("i", raw)[0]
    except (OSError, ValueError, AttributeError):
        return None


def parse_tcp_info(raw):
    if not raw:
        return {}
    fields, length = {}, len(raw)
    for name, (offset, fmt, size) in _TCP_INFO_FIELDS.items():
        if offset + size <= length:
            fields[name] = struct.unpack_from(fmt, raw, offset)[0]
    return fields


def _endpoint(getter):
    try:
        addr = getter()
    except (OSError, AttributeError):
        return ""
    if isinstance(addr, tuple) and len(addr) >= 2:
        return "%s:%s" % (addr[0], addr[1])
    return str(addr)


def _ip_of(endpoint):
    return endpoint.rsplit(":", 1)[0] if ":" in endpoint else endpoint


# ---------------------------------------------------------------------------
# CSV layout
# ---------------------------------------------------------------------------
NETWORK_COLUMNS = [
    "ts_start", "clock_offset_s", "node_id", "direction",
    "msg_type", "msg_class", "data_kind", "req_id", "client_id", "round",
    "local_addr", "peer_addr", "link_class", "wire_bytes",
    # send: serialization (pickle + framing) and sendall
    "t_serialize_s", "t_send_s",
    # recv: idle wait for the header, body transfer and unpickling
    "t_wait_s", "t_body_s", "t_deserialize_s",
    # send: wire_bytes / t_send_s; recv: body bytes / t_body_s
    "goodput_mbps",
    "rx_queued_at_header_bytes",
    # kernel state at the end of the call
    "tcp_rtt_us", "tcp_min_rtt_us", "tcp_rttvar_us", "tcp_rcv_rtt_us",
    "tcp_snd_cwnd", "tcp_snd_mss", "tcp_delivery_rate_mbps",
    "tcp_notsent_bytes_end", "tcp_unacked_segs_end",
    # kernel counters over the call
    "tcp_bytes_acked_delta", "tcp_bytes_received_delta", "tcp_retrans_segs_delta",
    "tcp_busy_us_delta", "tcp_rwnd_limited_us_delta", "tcp_sndbuf_limited_us_delta",
]

LINK_COLUMNS = [
    "first_seen_ts", "node_id", "hostname", "kernel", "local_addr", "peer_addr",
    "link_class", "iface", "gateway", "route_src", "nic_speed_mbps", "tcp_info_len",
]

TIMESYNC_COLUMNS = [
    "ts", "node_id", "local_addr", "peer_addr", "link_class", "rtt_s", "offset_s",
]


def _fmt(value, digits=6):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return "%.*f" % (digits, value)
    return value


def _delta(start, end, name):
    if name in start and name in end:
        return end[name] - start[name]
    return ""


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class NetRecorder(object):
    """Collects per-message records for one CommunicationModule instance."""

    def __init__(self, node_id):
        self.node_id = str(node_id)
        self.hostname = socket.gethostname()
        self.kernel = platform.release()

        self._queue = queue.SimpleQueue()
        self._run_id = None
        self._paths = {}
        self._files = {}
        self._writers = {}
        self._routes = {}
        self._links_seen = set()
        self._warned = set()

        self._enable_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # -- hot path -----------------------------------------------------------
    def record_send(self, sock, msg, wire_bytes, ts_start, clock_offset,
                    t_serialize, t_send, info_start, info_end):
        try:
            msg_type, req_id, client_id, round_idx = describe(msg)
            self._put(("message", {
                "direction": "send", "ts_start": ts_start, "clock_offset": clock_offset,
                "msg_type": msg_type, "req_id": req_id, "client_id": client_id, "round": round_idx,
                "local": _endpoint(sock.getsockname), "peer": _endpoint(sock.getpeername),
                "wire_bytes": wire_bytes, "t_serialize": t_serialize, "t_send": t_send,
                "info_start": info_start, "info_end": info_end,
            }))
        except Exception as e:  # recording must never break the send
            self._warn_once("record_send", f"[NetMetrics] send record skipped: {e}")

    def record_recv(self, sock, msg, wire_bytes, ts_start, clock_offset,
                    t_wait, t_body, t_deserialize, rx_queued, info_start, info_end):
        try:
            msg_type, req_id, client_id, round_idx = describe(msg)
            self._put(("message", {
                "direction": "recv", "ts_start": ts_start, "clock_offset": clock_offset,
                "msg_type": msg_type, "req_id": req_id, "client_id": client_id, "round": round_idx,
                "local": _endpoint(sock.getsockname), "peer": _endpoint(sock.getpeername),
                "wire_bytes": wire_bytes, "t_wait": t_wait, "t_body": t_body,
                "t_deserialize": t_deserialize, "rx_queued": rx_queued,
                "info_start": info_start, "info_end": info_end,
            }))
        except Exception as e:  # recording must never break the receive
            self._warn_once("record_recv", f"[NetMetrics] recv record skipped: {e}")

    def record_timesync(self, sock, ts, rtt, offset):
        try:
            self._put(("timesync", {
                "ts": ts, "local": _endpoint(sock.getsockname), "peer": _endpoint(sock.getpeername),
                "rtt": rtt, "offset": offset,
            }))
        except Exception as e:
            self._warn_once("record_timesync", f"[NetMetrics] time-sync record skipped: {e}")

    def _put(self, item):
        if self._run_id is None and self._queue.qsize() >= MAX_PENDING:
            self._warn_once("pending", f"[NetMetrics] {self.node_id}: recording never enabled "
                                       f"(no run id after {MAX_PENDING} messages); dropping records.")
            return
        self._queue.put(item)

    # -- lifecycle ----------------------------------------------------------
    def enable(self, run_id):
        """Starts writing once the run id is known. Later calls are no-ops."""
        if not run_id:
            return
        with self._enable_lock:
            if self._run_id is not None:
                if str(run_id) != self._run_id:
                    self._warn_once("run_id", f"[NetMetrics] {self.node_id}: already writing to run "
                                              f"{self._run_id}; ignoring run id {run_id}.")
                return
            run_id = str(run_id)
            base_dir = helpers.results_path(run_id, "network_metrics")
            os.makedirs(base_dir, exist_ok=True)
            stem = f"{self.node_id}_{self.hostname}_{run_id}"
            self._paths = {
                "message": (os.path.join(base_dir, f"{stem}_network.csv"), NETWORK_COLUMNS),
                "link": (os.path.join(base_dir, f"{stem}_links.csv"), LINK_COLUMNS),
                "timesync": (os.path.join(base_dir, f"{stem}_timesync.csv"), TIMESYNC_COLUMNS),
            }
            self._run_id = run_id
            self._thread = threading.Thread(target=self._run, name=f"net-metrics-{self.node_id}",
                                            daemon=True)
            self._thread.start()
            atexit.register(self.close)
            logger.info(f"[NetMetrics] {self.node_id}: recording to {base_dir}")

    def close(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=CLOSE_TIMEOUT_S)
        if self._run_id is None:
            return
        with self._write_lock:
            self._drain_locked()
            for f in self._files.values():
                try:
                    f.close()
                except OSError:
                    pass
            self._files.clear()
            self._writers.clear()

    # -- writer thread ------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            with self._write_lock:
                self._drain_locked()
            self._stop.wait(FLUSH_INTERVAL_S)

    def _drain_locked(self):
        wrote = False
        while True:
            try:
                kind, rec = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                if kind == "message":
                    self._write("message", self._message_row(rec))
                else:
                    self._write("timesync", self._timesync_row(rec))
                wrote = True
            except Exception as e:
                self._warn_once("write", f"[NetMetrics] {self.node_id}: failed to write a record: {e}")
        if wrote:
            for f in self._files.values():
                f.flush()

    def _write(self, kind, row):
        writer = self._writers.get(kind)
        if writer is None:
            path, columns = self._paths[kind]
            f = open(path, "a", newline="")
            writer = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
            if f.tell() == 0:
                writer.writeheader()
            self._files[kind] = f
            self._writers[kind] = writer
        writer.writerow(row)

    def _message_row(self, rec):
        msg_class, data_kind = classify(rec["msg_type"])
        if msg_class == "unclassified":
            self._warn_once(("unclassified", rec["msg_type"]),
                            f"[NetMetrics] message type not in the classification table: "
                            f"{rec['msg_type']} (add it to utils/communication/net_metrics.py)")

        start = parse_tcp_info(rec["info_start"])
        end = parse_tcp_info(rec["info_end"])
        link = self._link(rec["local"], rec["peer"], rec["ts_start"], rec["info_end"])

        send = rec["direction"] == "send"
        payload_bytes = rec["wire_bytes"] if send else rec["wire_bytes"] - HEADER_BYTES
        seconds = rec["t_send"] if send else rec["t_body"]
        goodput = payload_bytes * 8 / seconds / 1e6 if seconds > 0 else ""
        delivery = end.get("delivery_rate")

        return {
            "ts_start": _fmt(rec["ts_start"]),
            "clock_offset_s": _fmt(rec["clock_offset"]),
            "node_id": self.node_id,
            "direction": rec["direction"],
            "msg_type": rec["msg_type"],
            "msg_class": msg_class,
            "data_kind": data_kind,
            "req_id": rec["req_id"],
            "client_id": rec["client_id"],
            "round": rec["round"],
            "local_addr": rec["local"],
            "peer_addr": rec["peer"],
            "link_class": link["link_class"],
            "wire_bytes": rec["wire_bytes"],
            "t_serialize_s": _fmt(rec.get("t_serialize")),
            "t_send_s": _fmt(rec.get("t_send")),
            "t_wait_s": _fmt(rec.get("t_wait")),
            "t_body_s": _fmt(rec.get("t_body")),
            "t_deserialize_s": _fmt(rec.get("t_deserialize")),
            "goodput_mbps": _fmt(goodput, 3),
            "rx_queued_at_header_bytes": _fmt(rec.get("rx_queued")),
            "tcp_rtt_us": end.get("rtt", ""),
            "tcp_min_rtt_us": end.get("min_rtt", ""),
            "tcp_rttvar_us": end.get("rttvar", ""),
            "tcp_rcv_rtt_us": end.get("rcv_rtt", ""),
            "tcp_snd_cwnd": end.get("snd_cwnd", ""),
            "tcp_snd_mss": end.get("snd_mss", ""),
            "tcp_delivery_rate_mbps": _fmt(delivery * 8 / 1e6, 3) if delivery is not None else "",
            "tcp_notsent_bytes_end": end.get("notsent_bytes", "") if send else "",
            "tcp_unacked_segs_end": end.get("unacked", "") if send else "",
            "tcp_bytes_acked_delta": _delta(start, end, "bytes_acked") if send else "",
            "tcp_bytes_received_delta": "" if send else _delta(start, end, "bytes_received"),
            "tcp_retrans_segs_delta": _delta(start, end, "total_retrans"),
            "tcp_busy_us_delta": _delta(start, end, "busy_time") if send else "",
            "tcp_rwnd_limited_us_delta": _delta(start, end, "rwnd_limited") if send else "",
            "tcp_sndbuf_limited_us_delta": _delta(start, end, "sndbuf_limited") if send else "",
        }

    def _timesync_row(self, rec):
        link = self._link(rec["local"], rec["peer"], rec["ts"], None)
        return {
            "ts": _fmt(rec["ts"]),
            "node_id": self.node_id,
            "local_addr": rec["local"],
            "peer_addr": rec["peer"],
            "link_class": link["link_class"],
            "rtt_s": _fmt(rec["rtt"]),
            "offset_s": _fmt(rec["offset"]),
        }

    def _link(self, local, peer, ts, info_raw):
        """Link description of a connection; writes its links row the first time it is seen."""
        route = self._route(_ip_of(local), _ip_of(peer))
        key = (local, peer)
        if key not in self._links_seen:
            self._links_seen.add(key)
            self._write("link", {
                "first_seen_ts": _fmt(ts),
                "node_id": self.node_id,
                "hostname": self.hostname,
                "kernel": self.kernel,
                "local_addr": local,
                "peer_addr": peer,
                "link_class": route["link_class"],
                "iface": route["iface"],
                "gateway": route["gateway"],
                "route_src": route["route_src"],
                "nic_speed_mbps": route["nic_speed_mbps"],
                "tcp_info_len": len(info_raw) if info_raw else "",
            })
        return route

    def _route(self, local_ip, peer_ip):
        """
        loopback: both ends on this host; lan: peer reachable without a gateway;
        routed: through a gateway. A proxy shows up as its own IP and is told
        apart offline.
        """
        if peer_ip in self._routes:
            return self._routes[peer_ip]
        route = {"link_class": "unknown", "iface": "", "gateway": "", "route_src": "",
                 "nic_speed_mbps": ""}
        if peer_ip and (peer_ip == local_ip or peer_ip.startswith("127.")):
            route["link_class"] = "loopback"
        try:
            out = subprocess.run(["ip", "-o", "route", "get", peer_ip], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, universal_newlines=True,
                                 timeout=2).stdout
            tokens = out.split()

            def after(word):
                return tokens[tokens.index(word) + 1] if word in tokens[:-1] else ""

            route["iface"] = after("dev")
            route["gateway"] = after("via")
            route["route_src"] = after("src")
            if route["link_class"] != "loopback" and tokens:
                if tokens[0] == "local" or route["iface"] == "lo":
                    route["link_class"] = "loopback"
                else:
                    route["link_class"] = "routed" if route["gateway"] else "lan"
            if route["iface"] and route["iface"] != "lo":
                with open(f"/sys/class/net/{route['iface']}/speed") as f:
                    route["nic_speed_mbps"] = f.read().strip()
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        self._routes[peer_ip] = route
        return route

    def _warn_once(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(text)
