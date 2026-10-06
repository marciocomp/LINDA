#!/usr/bin/env bash
# Proxy 130.92.65.51 for the processes of 130.92.70.8 (site 1 and Tier 3)
#
# Sends connections that arrive at the proxy on the LINDA ports of 130.92.70.8 back to
# that host (DNAT) and rewrites the source to the proxy (SNAT), so the reply comes
# back through it. Without the SNAT the reply would go straight over loopback and
# the connection would break, because both ends are on the same machine.
#
# Ports: 10101, 10110 (Tier 3); 19100, 19103 (CM site1); 19400 (s1_t2_2)
# The Tier 1 ports are included: a client on this same host also goes through the
# proxy to reach its Cluster Manager.
# Both directions of each port are accepted explicitly, so a connection survives a
# moment when conntrack loses track of it.
#
# Run:  sudo bash proxy_65-51_to_70-8.sh
# Undo: sudo bash proxy_65-51_to_70-8.sh --flush-only     (leaves the proxy with no rules)
#
# Flushing wipes every rule of the three tables, including any ufw or docker rule,
# until those services are reloaded or the machine reboots.
set -euo pipefail

PROXY=130.92.65.51
HOST=130.92.70.8
# LINDA: 10101/10110 Tier 3, 19100-19103 CM site1, 19400 s1_t2_2
# Static HSFL baseline: 10102 orchestrator data, 20100-20500 chain stages
PORTS=(10101 10110 19100 19101 19102 19103 19400 10102 20100 20200 20300 20400 20500)

# --- clean slate -----------------------------------------------------------
iptables -F
iptables -F -t nat
iptables -F -t mangle
iptables -P FORWARD ACCEPT

if [ "${1:-}" = "--flush-only" ]; then
    echo "[proxy] tables flushed; no rule installed"
    exit 0
fi

# --- kernel settings -------------------------------------------------------
sysctl -w net.ipv4.ip_forward=1
# Keeps packets that fall outside the window conntrack expects instead of marking
# them INVALID; large transfers through a NAT are the usual reason for that.
sysctl -w net.netfilter.nf_conntrack_tcp_be_liberal=1
sysctl -w net.netfilter.nf_conntrack_max=1048576

# --- forwarding rules ------------------------------------------------------
iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

for PORT in "${PORTS[@]}"; do
    iptables -t nat -A PREROUTING  -d "$PROXY" -p tcp --dport "$PORT" -j DNAT --to-destination "$HOST:$PORT"
    iptables -t nat -A POSTROUTING -d "$HOST"  -p tcp --dport "$PORT" -j SNAT --to-source "$PROXY"
    iptables -A FORWARD -d "$HOST" -p tcp --dport "$PORT" -j ACCEPT
    iptables -A FORWARD -s "$HOST" -p tcp --sport "$PORT" -j ACCEPT
done

echo "[proxy] $PROXY -> $HOST on ports: ${PORTS[*]}"
iptables -t nat -L PREROUTING -n -v | grep DNAT
