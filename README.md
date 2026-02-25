# Enron SNA — Shadow Organization Intelligence

> Forensic social network analysis of the Enron email corpus. Maps the informal power structures, real working teams, and communication archetypes that operated beneath the company's official org chart — enriched by LLM-driven behavioral inference applied to actual email content.

---

## What This Is

The Enron email dataset contains roughly 500,000 emails from ~150 senior employees. The formal org chart told one story. The emails tell another.

This project runs a three-stage pipeline — parse, analyze, infer — to answer a deceptively simple question: **who actually held power, and how did information really flow?**

By combining graph-theoretic metrics (PageRank, betweenness centrality, Louvain community detection) with LLM analysis of raw email content, the pipeline produces a **Shadow Org Chart**: a map of real influence, informal roles, and hidden team structures rendered in an interactive 3D sphere visualization with deep per-node profiling.

---

## Pipeline Overview

```
emails.csv
    │
    ▼
[parse.py]  ──────────────────────────────  Filter & extract Enron emails
    │                                        Parallel chunk processing
    │
    ▼                                        enron_internal_only.csv
[process.py]  ────────────────────────────  Build communication graph
    │                                        Compute centrality metrics
    │                                        Detect communities (Louvain)
    │                                        Assign heuristic roles
    │
    ▼                                        sna_output/*.json + *.csv
[infer_power_roles.py]  ──────────────────  LLM forensic analysis per node
    │                                        French & Raven power bases
    │                                        Evidence quotes from real emails
    │
    ▼                                        llm_deep_enriched_nodes.json
[index.html]  ────────────────────────────  Interactive 3D sphere dashboard
                                             Community / Power / Role views
                                             Full per-node LLM profiles
```

---

## Stage 1 — `parse.py`

Reads the raw Enron dataset CSV and filters it down to internal Enron communications only, extracting structured header fields and email bodies.

### What it does

- Reads the raw corpus in chunks of 15,000 rows for memory efficiency
- Processes chunks in parallel using `ProcessPoolExecutor`
- Filters rows to keep only emails where the sender **or** recipient matches any Enron domain — handles all subdomains (`@enron.com`, `@hr.enron.com`, `@us.enron.com`, `@services.enron.com`, etc.) via the regex `[\w.+-]+@[\w.-]*enron\.com`
- Extracts `Date`, `From`, `To`, `Subject`, `X-From`, `X-To`, and the email body from each raw message
- Streams results directly to a CSV to avoid loading everything into memory at once

### Usage

```bash
python parse.py
```

**Input:** `emails.csv` (raw Enron corpus from Kaggle)
**Output:** `enron_internal_only.csv`

---

## Stage 2 — `process.py`

The core analysis pipeline. Reads the filtered CSV, builds a communication graph, computes network metrics, detects communities, assigns heuristic roles, and exports everything needed for the visualization and LLM inference steps.

### What it does

**Graph construction**

- Streams the CSV in chunks of 25,000 rows
- Parses each email into a directed edge (sender → recipient)
- Weights edges by `1/N` where N is the number of recipients — a 1:1 email counts more than a broadcast to 50 people
- Collapses directed edges into a weighted undirected graph for community detection, while also preserving a directed graph for in/out asymmetry analysis
- Tracks per-node counters: emails sent, emails received, unique contacts, months active

**Network metrics computed per node**

| Metric | Description |
|---|---|
| `pagerank` | Global influence — how much of the network's attention flows through this person |
| `betweenness` | Information brokerage — how often they sit on the shortest path between others |
| `eigenvector` | Prestige — being connected to influential people compounds your own score |
| `clustering` | Clique density — how tightly interconnected their local neighborhood is |
| `out_in_ratio` | Sent ÷ Received — a strong seniority signal; executives send more than they receive |
| `bridge_score` | Count of cross-community edges — high values flag cross-team connectors |
| `weighted_degree` | Total edge weight — accounts for communication frequency and exclusivity |

**Community detection**

Runs the Louvain algorithm on the largest connected component to partition the graph into clusters of people who communicate with each other more than with the rest of the network. These clusters reveal the *real* teams — often cutting across formal department lines.

**Heuristic role assignment**

Assigns a preliminary role label using structural signals alone, before the LLM refines it in Stage 3:

| Role | Classification logic |
|---|---|
| `Executive / Broker` | Top-10% PageRank AND top-25% betweenness |
| `Information Broker` | Top-25% betweenness only |
| `Broadcaster` | High sent volume; receives less than 30% of what they send |
| `Information Sink` | Receives 3× more than sends, high absolute volume |
| `Connector` | More than 50 unique contacts |
| `Regular Employee` | Everything else |

**LLM sample bank**

For each node, up to 15 representative email samples are collected using reservoir sampling — a statistically representative spread across their full communication history without loading every email into memory. These samples are what Stage 3 uses to do content-level analysis.

### Usage

```bash
python process.py
```

**Input:** `enron_internal_only.csv`
**Output:** `sna_output/` directory

| File | Contents |
|---|---|
| `graph_data.json` | Top-500 nodes and edges with all computed metrics |
| `llm_power_inputs.json` | Per-node metrics + sampled email snippets, ready for LLM |
| `communities.json` | Louvain community assignments, sizes, and top members |
| `stats.json` | Global corpus statistics |
| `top_nodes.json` | Top-50 nodes ranked by PageRank |
| `keywords.json` | Top-80 subject-line keywords by frequency |
| `timeline.json` | Monthly email volume over time |
| `degree_dist.json` | Degree distribution data for power-law analysis |
| `email_headers_metadata.csv` | Parsed header data for every valid email |

---

## Stage 3 — `infer_power_roles.py`

Sends each node's network metrics and sampled email snippets to an LLM for forensic behavioral analysis. This is where the Shadow Org Chart gets built.

### Why LLM inference is necessary

Graph metrics alone can't distinguish a powerful executive from a high-volume mailing list. A person who receives 3,000 emails might be the CEO or a passive distribution list. A person with low email volume might be the real decision-maker who delegates all correspondence to others. The LLM reads the actual words people wrote — and how they wrote them.

### What it does

The LLM is prompted as an elite professor of Organizational Behavior, applying established academic frameworks to each node's data. Two inputs are combined: structural metrics and raw email body snippets.

**Structured output schema enforced via Pydantic (`NodeProfile`)**

| Field | Description |
|---|---|
| `overall_power_index` | Integer 1–100. True organizational influence. 100 = CEO/top executive level |
| `french_and_raven_bases` | Scores 1–10 for Expert, Referent, Legitimate, and Coercive power |
| `inferred_informal_role` | Shadow org classification (e.g. *Hidden Decision Maker*, *Information Gatekeeper*, *Toxic Broker*) |
| `communication_style` | Linguistic fingerprint (e.g. *Directive and terse*, *Deferential and expansive*) |
| `evidence_quotes` | 2–3 verbatim email excerpts that prove the assigned power level |
| `comprehensive_analysis` | 150–200 word academic synthesis of network position × communication tone |

**The LLM is specifically instructed to:**

- Not be fooled by email volume — a passive Information Sink may receive 500 emails with zero real power
- Look for asymmetry: true executives typically have high out-degrees, low in-degrees, and write in short, directive sentences
- Distinguish between Information Brokers who *bridge* communities and those who *filter* them — both have high betweenness, but their organizational impact is opposite
- Apply French and Raven's Five Bases of Power to the text: does authority come from expertise, formal title, charisma, or coercion?

**Engineering design**

- Concurrent processing via `ThreadPoolExecutor` (default 8 workers, configurable via `INFER_WORKERS`)
- Each worker creates its own `OpenAI` client to avoid shared-state issues across threads
- Exponential backoff with up to 4 retry attempts per node on API failures
- Real-time checkpointing: after each node completes, the output is atomically written (`write to .tmp` → `os.replace`). If the run crashes at node 480 of 500, re-running automatically resumes from node 481

### Usage

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or a local proxy endpoint
export INFER_WORKERS=8                                # optional, default 8

python infer_power_roles.py
```

**Input:** `sna_output/llm_power_inputs.json`
**Output:** `sna_output/llm_deep_enriched_nodes.json`

---

## Stage 4 — `index.html` (Visualization Dashboard)

A single-file interactive dashboard. No build step, no server required — open directly in any modern browser. All data loads from GitHub raw URLs at runtime.

### Three view modes

Switch via the **Community / Power / Role** toggle in the left sidebar:

| Mode | What it shows |
|---|---|
| **Community View** | Nodes colored by Louvain cluster — the real teams |
| **Power View** | Nodes colored by LLM Power Index tier — the shadow hierarchy |
| **Role View** | Nodes colored by inferred informal archetype |

### Node detail panel

Click any node to open its full profile in the right sidebar:

- **Power Index** — large display with color-coded tier and progress bar
- **French & Raven Power Bases** — four colored bars for Expert, Referent, Legitimate, and Coercive power
- **Evidence Quotes** — verbatim email excerpts that justify the assigned power level
- **Academic Analysis** — the LLM's full synthesis paragraph
- **Network Metrics** — PageRank, Betweenness, Eigenvector, Clustering, Sent/Received, Out/In Ratio, Contacts
- **Neighbours** — top connected nodes by edge weight, all clickable

### Power Index color scale

| Color | Tier | Range |
|---|---|---|
| 🔴 Red | Elite | 80–100 |
| 🟡 Amber | Senior | 60–79 |
| 🟢 Green | Mid-level | 40–59 |
| 🔵 Cyan | Junior | 20–39 |
| ⬛ Gray | Staff / Peripheral | 0–19 |

### Controls reference

**Toolbar buttons:**

| Button | Action |
|---|---|
| `Select` | Click nodes to inspect |
| `Lasso` | Draw a rectangle to multi-select |
| `Pan` | Drag to pan, scroll to zoom |
| `Labels` | Toggle node name labels |
| `Ego` | Show only a node's 1-hop neighborhood |
| `Path` | Find the shortest communication path between two nodes |
| `PNG` | Export current view as a high-res image |
| `JSON` | Export the currently visible nodes and edges |

**Keyboard shortcuts:**

| Key | Action |
|---|---|
| `L` | Lasso mode |
| `P` | Pan mode |
| `E` | Ego network (requires 1 selected node) |
| `R` | Rebuild graph |
| `F` | Fit to view |
| `H` | Toggle labels |
| `Esc` | Clear selection / exit ego mode |
| `Shift+Click` | Add node to selection |
| `Double-click` | Pin / unpin node |
| `Alt+Click` | Start shortest-path from that node |
| `?` | Keyboard help overlay |

**Left sidebar filters:**

- **Min Edge Weight** — filter out low-frequency communication links
- **Max Nodes** — limit visible nodes, sorted by the current size metric
- **Sphere Radius** — scale the sphere layout in or out
- **Spin Speed** — control auto-rotation speed
- **Node Size by** — encode size as PageRank, LLM Power Index, Betweenness, Degree, Sent, or Received

**Right sidebar tabs:**

| Tab | Contents |
|---|---|
| **Overview** | Community size doughnut, Power Index histogram, subject keyword cloud |
| **Analytics** | Email volume timeline, degree distribution (log-log), power tier breakdown, community stats table |
| **Selected** | Full LLM profile for one node, or aggregate stats for a multi-selection |

---

## Repository Structure

```
enron-sna/
├── parse.py                          # Stage 1: filter raw corpus → internal CSV
├── process.py                        # Stage 2: graph, metrics, communities, roles
├── infer_power_roles.py              # Stage 3: LLM forensic power-level analysis
├── index.html                        # Stage 4: interactive visualization dashboard
└── sna_output/
    ├── graph_data.json               # Nodes + edges with all computed metrics
    ├── llm_power_inputs.json         # Metrics + email samples ready for LLM
    ├── llm_deep_enriched_nodes.json  # Final LLM-enriched profiles (500 nodes)
    ├── communities.json              # Louvain community assignments and stats
    ├── stats.json                    # Global corpus statistics
    ├── keywords.json                 # Top subject-line keywords
    ├── timeline.json                 # Monthly email volume
    ├── degree_dist.json              # Degree distribution
    ├── top_nodes.json                # Top-50 nodes by PageRank
    └── email_headers_metadata.csv    # Parsed header data for all valid emails
```

---

## Setup

```bash
pip install pandas numpy networkx python-louvain tqdm openai pydantic
```

| Package | Used in | Purpose |
|---|---|---|
| `pandas` | parse, process | CSV I/O and chunk streaming |
| `numpy` | process | Vectorised metric computation |
| `networkx` | process | Graph construction and centrality algorithms |
| `python-louvain` | process | Louvain community detection (import as `community`) |
| `tqdm` | process, infer | Progress bars |
| `openai` | infer | LLM API calls with structured output parsing |
| `pydantic` | infer | Schema enforcement for `NodeProfile` |
| `d3` v7 | index.html | Graph rendering and 3D sphere layout (CDN) |
| `chart.js` v4 | index.html | Analytics charts (CDN) |

---

## Running the Full Pipeline

```bash
# Stage 1 — filter the raw corpus
python parse.py

# Stage 2 — build graph and export all datasets
python process.py

# Stage 3 — run LLM inference on all 500 nodes
export OPENAI_API_KEY="sk-..."
python infer_power_roles.py

# Stage 4 — open the dashboard (no server needed)
open index.html
```

The dashboard works without completing Stage 3. Stages 1 and 2 produce all the data needed for the Community and Role views. Stage 3 enriches each node with deep behavioral analysis and is what unlocks the Power View, French & Raven bars, and evidence quotes.

---

## Key Findings

**High Power Index + High Betweenness = Information Broker.** These individuals sit at the crossroads of otherwise disconnected groups. Often mid-level managers rather than C-suite, but removing them would fragment the network. The LLM frequently assigns them high Legitimate and Expert power scores.

**High Out/In Ratio + Low Betweenness = Broadcaster.** Sends constantly but doesn't bridge communities. Typically project leads or domain specialists pushing updates outward. High volume does not mean high power.

**High Received + Low Sent = Information Sink.** Could be a distribution list, an executive's assistant, or someone CC'd reflexively. The LLM reliably assigns these nodes low power indices regardless of their email volume — a distinction that pure metric analysis consistently misses.

**Community ≠ Department.** Louvain clusters frequently cross formal reporting lines, revealing working groups that formed organically around projects, crises, or shared information needs. These are the gaps between the official org chart and how work actually got done.

---

## Data

The Enron email dataset is a public domain corpus released by the Federal Energy Regulatory Commission during its investigation into the company's collapse. It is widely used in network science, NLP, and organizational behavior research. The raw `emails.csv` is available from [Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset).

---

## Author

Built by **Mayank** — combining graph theory, community detection, and LLM behavioral inference to map the hidden power structures of one of history's most studied corporate collapses.