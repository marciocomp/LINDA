# @author: Marcio Lopes

# --- Model Constraints (Gemma-2B LoRA) ---
# Defined in Sec V.B: "footprint of approx. 10-14GB"
# We set 12GB to ensure Site 2 (6GB) is a Straggler and Site 3 (24GB) is Strong.
MODEL_MEMORY_REQ = 12.0  # GB
TOTAL_LAYERS = 32        # Number of transformer layers in Gemma-2B

# --- Network Emulation (WAN) ---
# Defined in Sec V.A: "simulate a low-bandwidth (e.g., 100 Mbps)"
WAN_BANDWIDTH_MBPS = 100.0
WAN_LATENCY_MS = 80      # Round-Trip Time (RTT)
# Default Bandwidth assumption (in MB/s) if not provided.
# 100 Mbps ~ 12.5 MB/s (WAN), 1 Gbps ~ 125 MB/s (LAN)
DEFAULT_BANDWIDTH = 50.0
#

BW_LAN = 125.0  # ~1 Gbps (Device <-> Edge)
BW_WAN = 12.5   # ~100 Mbps (Edge <-> Cloud)

# --- Profiling Config ---
PROFILE_INTERVAL = 10    # Seconds between resource checks
