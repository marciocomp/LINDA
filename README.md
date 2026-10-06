# LINDA

**L**earning on **I**nterconnected **N**odes via **D**ynamic **A**llocation — a
memory-aware framework for distributed training of Large Language Models across
a Multi-Cluster Continuum.

LINDA implements **2HSFL** (Horizontal and Hierarchical Split Federated
Learning): model segments are mapped onto a three-tier topology, chained
horizontally across the peer nodes of a local compute pool, and offloaded
hierarchically to a remote aggregation pool only when a cluster cannot hold its
share. Placement is decided from the memory and throughput each device reports,
and recomputed between rounds, so a node that loses memory to a competing
process is relieved before it fails.

This is a research prototype, not a simulator. Every tier is a separate
operating-system process communicating over real TCP sockets, and node
capabilities are read from the devices themselves — free VRAM from the CUDA
runtime and throughput from a timed matmul — rather than declared in a file.

> **Usage instructions are in progress** — see [Usage](#usage).

## Packages

### `frameworkLINDA/`

The framework itself: one module per tier of the 2HSFL topology.

| module | role |
|---|---|
| `tier_1/node_agent.py` | Data source. Holds the raw data, runs the first layers locally and emits the smashed activations. The raw dataset never leaves this process. |
| `tier_2/cluster_manager.py` | Head of a compute pool. Control plane of its cluster and first server of the chain: receives the allocation, executes its segment and routes to the next hop. |
| `tier_2/compute_node.py` | Compute worker. Executes one segment of the chain and forwards to its successor. |
| `tier_3/orchestrator.py` | Aggregation pool. Decides the allocation, absorbs the residual segments of straggler clusters, aggregates the model and evaluates it. |

### `runLINDA/`

Everything that drives an experiment: the runners of each tier, the topology
and hardware descriptions, the baselines and the analysis scripts.

| path | content |
|---|---|
| `case_studies/cds/cds_tier_{1,2,3}/` | Entry points. One runner per tier, launched from its own directory. |
| `case_studies/cds/*.json` | Topology, link and hardware descriptions of the deployment. |
| `case_studies/cds/baselines/` | Standard SFL, Static HSFL and centralized baselines, standalone scripts. |
| `case_studies/drift_injector.py` | Competes for the VRAM of a chosen node, to exercise runtime re-allocation. |
| `case_studies/*.py` | Analysis: latency and network breakdowns, convergence, viability plots. |

### `utils/`

Shared infrastructure, used by every tier.

| path | content |
|---|---|
| `communication/` | Base class of all tiers: pickled objects over TCP, separate control and data sockets, clock synchronization, per-message network records. |
| `models/` | Model shards. `gemma_wrapper.py` prunes the layers outside a node's range and rescales the incoming activations; `resnet50.py` and `vgg19.py` are the vision counterparts. |
| `datasets/` | Alpaca, CIFAR-10 and ImageNet loaders, with IID and Dirichlet partitioning. |
| `standardTuples/` | The payloads exchanged between tiers: allocation rows, activations, gradients and resource vectors. |
| `helpers.py` | Memory accounting per model block, metric and snapshot writers, optimizer-moment carrying across a re-allocation. |
| `model_factory.py` | The single entry point for building a shard. |

### `tpn_artifacts/`

The Timed Petri Net that specifies the orchestration workflow, and the reports
produced from it.

| file | content |
|---|---|
| `linda_tpn_n3.xml` | the model in PNML, one client's pipeline with three Compute Workers: 8 places, 10 transitions, a single token at `p_S`, and an arc returning `p_Sigma` to `p_S` so that the analysis spans successive rounds |
| `classification.html` | the net is a state machine: every transition has one input and one output place |
| `invariant_analysis.html` | one minimal P-invariant of unit weight, giving `sum M(p) = 1`, and four minimal T-invariants, one per operational configuration of LINDA |
| `state_space_analysis.html` | bounded, safe, no dead markings |
| `incidence_and_marking.html` | the incidence matrices and the initial marking |
| `README.txt` | what each place and transition stands for, and which file supports which claim |

The analysis was run with **PIPE v4.3.0**, the Platform Independent Petri Net
Editor developed at Imperial College London. Its design is described in
N. J. Dingle, W. J. Knottenbelt and T. Suto, "PIPE2: a tool for the performance
evaluation of generalised stochastic Petri Nets", *ACM SIGMETRICS Performance
Evaluation Review*, vol. 36, no. 4, pp. 34–39, 2009
([doi:10.1145/1530873.1530881](https://doi.org/10.1145/1530873.1530881)), and
the tool is distributed from
[sourceforge.net/projects/pipe2](https://sourceforge.net/projects/pipe2/).

The reachability graph is not among the files: PIPE draws it in a window of its
own and writes no file, so it has to be generated from the model with the tool.
`tpn_artifacts/README.txt` says how.

## Requirements

```bash
pip install -r utils/requirements.txt
```

Two files have to be configured before the first run:
`utils/huggingface_token.py`, which Gemma needs, and `utils/kaggle.json`, which
the dataset downloads need. Both are tracked as templates and carry their own
instructions, including how to keep your credentials out of the commits.

Gemma-2B is distributed by Google under its own
[terms of use](https://ai.google.dev/gemma/terms), which you accept when you
download the weights.

Every reported number comes from Python 3.8.20, PyTorch 2.4.1+cu121 and
transformers 4.46.3, on NVIDIA A40, RTX 4090 and RTX 3090 cards with Intel i9
and AMD Threadripper hosts as the CPU-only data sources. Later runs on Python
3.10 with PyTorch 2.9 and transformers 4.57 also completed, so the code is not
tied to that stack, but it is the one behind the results. Every machine of a
deployment needs the repository and the dependencies; the processes share
nothing but the network.

## Usage

### 1. Describe your deployment

A run is defined by a **case study**: a directory under `runLINDA/case_studies/`
that describes your machines and holds one launch directory per tier. The case
study reported in the paper is `cds`; start from it.

```bash
cp -r runLINDA/case_studies/cds runLINDA/case_studies/<my_case>
```

Rename the copy's topology file and its three tier directories after the case
study, so that the layout reads:

```
runLINDA/case_studies/<my_case>/
  network_config_<my_case>.json    your topology: the one file you have to write
  network_links.json               the links between the nodes declared above
  <my_case>_tier_1/                launch directory of the Node Agents
  <my_case>_tier_2/                launch directory of the Cluster Managers and Compute Workers
  <my_case>_tier_3/                launch directory of the Orchestrator
```

Nothing else about the hardware is read from a file. The memory and the
throughput of every Tier 2 and Tier 3 device are measured by the device itself
at startup, and again between rounds when runtime re-allocation is enabled, so
`hardware_specs.json` — which earlier versions consulted — is no longer read and
can be left behind.

#### The topology file

Three tiers per site, each a named object of nodes:

```json
{
  "proxies": { "130.92.70.8": "130.92.65.51" },
  "sites": {
    "site1": {
      "description": "Lead cluster, orchestration",
      "tier3": {
        "leader":   { "id": "s1_t3_1", "device": "cuda:2", "hostname": "server_a40_5",
                      "ip": "130.92.70.8", "port": [10101, 10110] }
      },
      "tier2": {
        "server_1": { "id": "s1_t2_1", "device": "cuda:1", "hostname": "server_a40_3",
                      "ip": "130.92.70.8", "port": [19100, 19101, 19102, 19103] },
        "server_2": { "id": "s1_t2_2", "device": "cuda:0", "hostname": "server_a40_2",
                      "ip": "130.92.70.8", "port": [19400] }
      },
      "tier1": {
        "device_1": { "id": "s1_t1_1", "device": "cpu", "hostname": "host_amd",
                      "ip": "130.92.70.22", "port": [8100] }
      }
    },
    "site2": { "tier3": {}, "tier2": { "...": {} }, "tier1": { "...": {} } }
  }
}
```

Node identifiers follow `s{site}_t{tier}_{index}`, and the tiers are the ones of
the paper: Tier 1 holds the data, Tier 2 is the compute pool, Tier 3
orchestrates and aggregates. Four assumptions of the code are worth knowing
before you write the file:

| assumption | consequence |
|---|---|
| the site that owns Tier 3 is named `site1` | the Tier 2 processes look the Orchestrator up at the fixed path `sites.site1.tier3.leader`; every other site declares `"tier3": {}` |
| the **first** entry of a site's `tier2` is its Cluster Manager | it is taken with `next(iter(...))`, so the order of the keys in the JSON object decides which node heads the cluster |
| a Cluster Manager needs four ports, a Compute Worker one | the indices are positional, see below |
| the number of sites and of GPUs per pool is free | the allocator assumes no particular shape: one site or many, one card per pool or several |

Ports are read by index, not by name:

| owner | index | accepts |
|---|---|---|
| Orchestrator | 0 | control plane |
| | 1 | data plane |
| Cluster Manager | 0 | the Compute Workers of its cluster, control |
| | 1 | the Node Agents of its site, control |
| | 2 | the Node Agents of its site, data |
| | 3 | the Compute Workers of its cluster, data |
| Compute Worker | 0 | its predecessor in the chain |

A Tier 1 entry is read for its `ip` and its `device` only; a Node Agent dials out
and binds nothing, so its `port` is declared for symmetry and never used.

`network_links.json` describes the resource graph the Orchestrator reasons over:
one object per edge, `{"u", "v", "type", "bandwidth_mbps", "latency_ms"}`, where
`u` and `v` are node identifiers of the topology. It is read from the fixed
relative path `../network_links.json`, so it belongs next to the topology file.

The optional `proxies` section is keyed by **host**, not by site. Two processes
on one host would exchange their tensors over loopback, which no network record
can measure as a real hop; declaring that host sends the traffic out to a
forwarding address on another machine and back
(`runLINDA/case_studies/cds/proxy_*.sh` installs the DNAT and SNAT rules). One
entry per co-located host, whatever the number of sites, and hops between hosts
are always direct. A topology that declares no `proxies` is proxied nowhere, so
a deployment on one machine runs by pointing `--lab` at a topology without the
section.

### 2. Register the case study in the runners

Each tier directory holds the runner of its tier and the configuration builder
that reads the topology for it:

| directory | files |
|---|---|
| `<my_case>_tier_1/` | `run_node_agent.py`, `config_node_agent.py` |
| `<my_case>_tier_2/` | `run_cluster_manager.py`, `config_cluster_manager.py`, `run_compute_node.py`, `config_compute_node.py` |
| `<my_case>_tier_3/` | `run_leader.py`, `config_leader.py` |

Which topology file a process loads is decided by `--lab`, and the mapping is
spelled out separately in each of the four runners. Add your case study to all
four:

```python
if lab == "<my_case>":
    topology_json_path = "../network_config_<my_case>.json"
```

`run_node_agent.py` picks the dataset directory in the same branch, so there it
takes one line more.

Every process is launched **from its own tier directory**, and the paths are what
make that necessary: the runners reach the repository root with
`sys.path.append('../../../../')`, resolve the topology as
`../network_config_*.json`, and write their output under `../../results/`. A
tier directory therefore has to sit exactly at
`runLINDA/case_studies/<my_case>/<tier_dir>/`. The results are the exception:
their root is resolved from the installation, not from the working directory, so
every case study writes into `runLINDA/case_studies/results/{run_id}/`, one tree
per run.

### 3. Prepare the data

Only Tier 1 touches the data. The Node Agents hold it, partition it and run the
first layer; neither the compute pool nor the Orchestrator ever loads a dataset.

| `--dataset` | source | how it arrives |
|---|---|---|
| `alpaca` | `tatsu-lab/alpaca` on Hugging Face | fetched and cached on first use |
| `cifar10` | torchvision | fetched on first use |
| `imagenet` | ImageNet-mini, from Kaggle | **manually**: there is no auto-download |

ImageNet has to be extracted into `train/` and `val/` directories of class
folders before the first run; the loader stops with that instruction if it is
not. Fetching it is a manual step, outside the framework, and
`utils/kaggle.json` is where the Kaggle credentials for it are kept.

Each Tier 1 host needs its own copy, and the path comes from the `--lab` branch
of `run_node_agent.py`, beside the topology path. `--data_dir` is parsed and
then ignored, so that branch is the place to edit.

`--new_size` truncates the dataset before it is partitioned: a fraction below
`1.0`, an absolute number of samples at or above it.

### 4. Run an experiment

One process per node, each launched from its own tier directory. Tier 3 binds
first; the lower tiers retry ten times before giving up, so they can be started
slightly early.

```bash
# Tier 3 — from <my_case>_tier_3/
python run_leader.py --lab <my_case> --model_name google/gemma-2b --model_type llm \
    --rounds 15 --batch_size 8 --lr 5e-6 --alpha 1.0 --mu 0.01 --fedprox_cpu --seed 20260311

# Tier 2 — from <my_case>_tier_2/, one Cluster Manager per site
python run_cluster_manager.py --lab <my_case> --site site1 --id s1_t2_1
python run_cluster_manager.py --lab <my_case> --site site2 --id s2_t2_1

# Tier 2 — from <my_case>_tier_2/, every remaining node of every pool
python run_compute_node.py --lab <my_case> --site site1 --id s1_t2_2

# Tier 1 — from <my_case>_tier_1/, one per data source
python run_node_agent.py --lab <my_case> --site site1 --id s1_t1_1 --dataset alpaca --new_size 0.1
```

Only the Orchestrator takes model and hyperparameter flags. Everything else
travels downhill over the control plane, so the other three runners take just
`--lab`, `--site` and `--id` — plus, for a Node Agent, what its data looks like:

| flag | default | |
|---|---|---|
| `--model_name` | `google/gemma-2b` | `gemma*` or `resnet50`; nothing else is built |
| `--model_type` | `llm` | `llm` or `vision` |
| `--rounds` | `15` | global rounds |
| `--batch_size` | `16` | |
| `--lr` | `5e-6` | decayed every round, by a factor fixed per model type |
| `--alpha` | `1.0` | Dirichlet concentration of the partition |
| `--mu` | `0` | `0` aggregates with FedAvg, anything above it with FedProx |
| `--fedprox_cpu` | off | keeps the global copy FedProx needs on the host, not on the card |
| `--seed` | `20260311` | |
| `--realloc` | off | re-plan the placement at a round boundary when the measured resources call for it |
| `--realloc_min_rounds` | `1` | consecutive rounds that must agree before a change is acted on |
| `--dataset` | `alpaca` | on a Node Agent, which data it holds; on the Orchestrator, only the class count of a vision run |
| `--new_size` | `0.9` | Node Agent only: truncates the dataset before partitioning — a fraction below `1.0`, a sample count above it |

`--epochs` and `--straggler` are parsed and go no further: a cluster becomes a
straggler because the allocation sends its tail to Tier 3, never because of a
flag.

#### How many sites

Any number, from one upward, **provided Tier 3 lives in `site1`**. That is the
only position the topology fixes; everything else — how many sites, how many
GPUs in a pool, how many data sources behind a cluster — is read from the file
and carried through the allocator, which iterates over the sites it finds.

Two consequences at launch. The Orchestrator waits for exactly one Cluster
Manager per site declared in the topology, so a site you declare but never start
leaves the run blocked before training; and a site whose `tier1` is empty is
skipped by the allocator, since there is no client to serve.

#### What a run produces

The Orchestrator mints the `run_id` (`YYYYMMDDHHMM`) and pushes it to every
node, so all processes write into one tree under
`runLINDA/case_studies/results/{run_id}/`:

| directory | content |
|---|---|
| `configs/` | the configuration each node received |
| `performance_metrics/` | the per-node timers of every phase of every round |
| `network_metrics/` | one row per message and one per TCP connection, at both ends |
| `snapshots/` | VRAM and RAM around each batch |
| `training_logs/`, `metrics/` | per-batch losses and the global evaluation |
| `reallocation/` | decisions, device profiles and discovery times — only with `--realloc` |
| `failures/` | written only if a node died, with the device state at that instant |

A node whose training loop raises writes its row to `failures/` and exits with
code **17**, rather than leaving the rest of the site waiting on a forward pass
that is not coming.

### 5. Runtime re-allocation

Without `--realloc` the placement is computed once, at startup, and never
revisited — the behaviour the experiments compare against. With it, the
Orchestrator asks every site for a fresh reading after each evaluation, re-runs
the heuristic, and when the result differs from the placement in force it
distributes the new one together with the parameters of every range that
changed, cut from the weights the round has just averaged. A candidate that no
longer fits, or that would change which nodes form a chain, is refused and the
placement in force stays. `--realloc_min_rounds` sets how many consecutive
rounds must ask for the same change before it is acted on.

Three files appear under `reallocation/`:

| file | one row per |
|---|---|
| `decisions_*.csv` | round boundary: what moved, what the candidate would have moved, and why |
| `profiles_*.csv` | node per round: the measured vector beside the one the placement in force was computed from |
| `discovery_*.csv` | site: where the discovery time went |

#### Making the resources move

`drift_injector.py` competes for a card, so that a run's resource state really
changes. It resolves its own paths, so it runs from anywhere; one process on
each host that holds a target card.

```bash
# take 7 GiB of cuda:0 and hold it until stopped
python runLINDA/case_studies/drift_injector.py --device cuda:0 --gib 7

# take whatever leaves 5 GiB free, release after ten minutes
python runLINDA/case_studies/drift_injector.py --device cuda:0 --leave-free 5 --hold-seconds 600
```

`--gpu-load N` also competes for the card's arithmetic, running kernels N% of
the time — a duty cycle, since a GPU is either running one or not, and what
`nvidia-smi` reports as utilization. `--start-at` and `--trigger-file` delay the
take, and `--dry-run` says what would be taken without touching anything.

For a campaign where several seeds have to suffer the same drift at the same
moments, `--schedule` drives the injections from the run's own progress instead
of from the clock:

```bash
# the same command on every host that holds one of the target cards
python runLINDA/case_studies/drift_injector.py --schedule --run-id 202609281530 --gib 7 5 15
```

The `SCHEDULE` constant at the top of the file fixes the shape — which node is
squeezed during which round, and what is released first — and `--gib` gives one
amount per step. Each host resolves the targets through the topology and runs
only the steps whose card is its own, which is why one command line covers the
whole deployment. The injector reads the run's own metrics to know which round
it is in (`--watch`), and if it finds that marker too long after it was written
(`--max-lag`) it releases everything and aborts, so that a seed whose step
missed its moment is missing rather than mislabelled.

## How to cite

The paper describing LINDA is under review. **Reference TBA.**

## Data repository

The execution logs behind every result reported in the paper are deposited in
REDU, the research data repository of the University of Campinas, under
[10.25824/redu/N72W9U](https://doi.org/10.25824/redu/N72W9U).

```bibtex
@data{Lopes2026LINDAdata,
    author = {Lopes, Marcio Moraes and Samikwa, Eric and Braun, Torsten and Bittencourt, Luiz Fernando},
    publisher = {Repositório de Dados de Pesquisa da Unicamp },
    title = {{Experimental results from Horizontal and Hierarchical Split Federated Learning of LLMs on a multi-cluster continuum testbed}},
    year = {2026},
    version = {DRAFT VERSION},
    doi = {10.25824/redu/N72W9U},
    url = {https://doi.org/10.25824/redu/N72W9U}
}
```

The deposit holds 50 executions grouped by experiment, with the per-node
performance timers, the per-message network records, the memory snapshots, the
training logs and the allocation decisions of each run. The data are licensed
under CC BY-NC 4.0, separately from the code.

## License

This software is released under the **GNU General Public License v3.0**. See
[COPYING](COPYING) for the full text.

## Acknowledgment

This work was partially funded by Grant #2019/26702-8, São Paulo Research
Foundation (FAPESP).
