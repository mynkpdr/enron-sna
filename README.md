# Enron SNA — Social Network Analysis Dashboard

An end-to-end social network analysis pipeline and interactive 3-D visualization dashboard for [The Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) (~500 K emails, ~1.3 GB).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Python Pipeline — `main.ipynb`](#python-pipeline--mainipynb)
   - [Cell 1 — Imports & Configuration](#cell-1--imports--configuration)
   - [Cell 2 — Streaming CSV Parser](#cell-2--streaming-csv-parser)
   - [Cell 3 — Edge Aggregation](#cell-3--edge-aggregation)
   - [Cell 4 — Graph Construction](#cell-4--graph-construction)
   - [Cell 5 — Centrality Metrics](#cell-5--centrality-metrics)
   - [Cell 6 — Community Detection](#cell-6--community-detection)
   - [Cell 7 — Role Inference](#cell-7--role-inference)
   - [Cell 8 — Temporal Analysis & Keywords](#cell-8--temporal-analysis--keywords)
   - [Cell 9 — JSON Export](#cell-9--json-export)
5. [Dashboard — `index.html`](#dashboard--indexhtml)
   - [Header Bar](#header-bar)
   - [Left Sidebar — Controls](#left-sidebar--controls)
   - [Main Canvas — Network Sphere](#main-canvas--network-sphere)
   - [Toolbar Buttons](#toolbar-buttons)
   - [Keyboard Shortcuts](#keyboard-shortcuts)
   - [Right Sidebar — Analytics Panels](#right-sidebar--analytics-panels)
   - [Tooltip](#tooltip)
6. [Output Files](#output-files)
7. [Algorithms & Design Decisions](#algorithms--design-decisions)
8. [Dependencies](#dependencies)

---

## Project Overview

This project constructs an **undirected weighted social graph** from the Enron email metadata, applies social network analysis (SNA) algorithms to uncover communication roles and communities, then renders the result as an interactive 3-D sphere in the browser — no server required after the initial data generation step.

**Key design goals:**

| Goal | How it is achieved |
|---|---|
| Low RAM usage | CSV is streamed in 20 K-row chunks; lists replace DataFrames during accumulation |
| Speed | Header-only regex parser (skips email body); betweenness uses a 200-node sample |
| Insight | Louvain communities, PageRank, betweenness, role inference, temporal timeline |
| Interactivity | Pure vanilla JS + D3 v7 + Chart.js; no build step, opens directly in a browser |

---

## Repository Structure

```
.
├── emails.csv                  # Raw Enron corpus (not tracked by git)
├── main.ipynb                  # Python analysis pipeline (Jupyter Notebook)
├── index.html                  # Interactive SNA dashboard (self-contained)
├── sna_output/                 # Generated data files (served by the dashboard)
│   ├── graph_data.json         # Top-500 nodes + edges
│   ├── top_nodes.json          # Top-30 nodes by PageRank
│   ├── communities.json        # Community summaries
│   ├── timeline.json           # Monthly email volume
│   ├── keywords.json           # Top-80 subject keywords
│   ├── degree_dist.json        # Degree distribution
│   ├── stats.json              # Global graph statistics
│   └── email_headers_metadata.csv  # Per-email metadata (date, from, subject)
├── TASK.md                     # Original project brief
└── README.md                   # This file
```

> **Note:** `emails.csv` is excluded from version control (see `.gitignore`).  
> Download it from [The Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) and place it in the project root.

---

## Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy networkx python-louvain tqdm
```

### 2. Run the pipeline

Open `main.ipynb` in Jupyter (or VS Code) and **Run All**.  
This creates all files in `sna_output/`.

### 3. Serve and open the dashboard

```bash
python -m http.server 8000
```

Then open [http://localhost:8000/index.html](http://localhost:8000/index.html) in your browser.  
*(The dashboard fetches files from `sna_output/` via `fetch()`, which requires a local HTTP server or a browser that allows local file access.)*

---

## Python Pipeline — `main.ipynb`

### Cell 1 — Imports & Configuration

**Purpose:** Sets up the environment, defines regex patterns, and configures paths.

| Variable | Default | Description |
|---|---|---|
| `EMAILS_CSV` | `"emails.csv"` | Path to the raw Enron CSV |
| `OUTPUT_DIR` | `"sna_output"` | Directory where JSON outputs are written |
| `CHUNK_SIZE` | `20_000` | Rows per CSV chunk; reduce if RAM is limited |
| `ENRON_RE` | regex | Matches only `@enron.com` addresses to filter out external emails |

**Helper functions:**

- **`fast_parse(raw: str)`** — Extracts `(sender, recipients, date_str, subject)` from a raw email message string using five pre-compiled regexes (`_FROM_RE`, `_TO_RE`, `_CC_RE`, `_DATE_RE`, `_SUBJ_RE`). Only scans the first 2 000 characters (headers), making it ~15× faster than the stdlib `email.message_from_string()` which also parses the body. Returns `None` if the sender is not an Enron address or if there are no Enron recipients.

- **`parse_date_ymd(s: str)`** — Converts an RFC-2822 date string to a `(YYYY-MM-DD, YYYY-MM)` tuple. Handles 2-digit years and malformed dates gracefully.

---

### Cell 2 — Streaming CSV Parser

**Purpose:** Reads `emails.csv` in chunks, parses each email header with `fast_parse`, and accumulates raw edge data into flat Python lists for memory efficiency.

**How the tie-strength weighting works:**  
For each email sent to `N` recipients (To + Cc), every resulting edge gets a fractional weight of `1/N`. This prevents mass broadcast emails from dominating the graph. Weights are later summed per node-pair during aggregation.

**Data collected per email:**

| List | Content |
|---|---|
| `src_list` / `tgt_list` | Sender and each recipient |
| `w_list` | Edge weight (`1/N`) |
| `rc_list` | Raw recipient count (always 1 per occurrence) |
| `dc_list` | 1 if the email was sent to a single recipient (direct) |
| `bc_list` | 1 if the email was sent to more than 5 recipients (broadcast) |
| `sent_ctr` | Counter of emails sent per person |
| `recv_ctr` | Counter of emails received per person |
| `month_sets` | Set of active months per person |
| `subj_list` | All subject lines (for keyword extraction) |
| `headers_meta` | List of `{date, from, to_cc_count, subject}` dicts |

`gc.collect()` is called after each chunk to prevent RAM accumulation from lingering references.

---

### Cell 3 — Edge Aggregation

**Purpose:** Converts the flat edge lists into a pandas DataFrame and aggregates parallel and reverse edges into a single undirected edge per pair.

To canonicalise direction, any edge where `src > tgt` (lexicographically) has its endpoints swapped, so each pair appears exactly once regardless of who sent to whom.

`groupby(["src", "tgt"])` then computes:

| Column | Meaning |
|---|---|
| `weight` | Sum of `1/N` fractional weights |
| `raw_count` | Total number of email occurrences |
| `direct_count` | Emails sent directly (1-to-1) |
| `broadcast_count` | Emails sent to 6+ recipients |

---

### Cell 4 — Graph Construction

**Purpose:** Builds a NetworkX `Graph` from the aggregated edge DataFrame and attaches per-node attributes.

```python
G = nx.from_pandas_edgelist(df_agg, source="src", target="tgt",
                             edge_attr=["weight", "raw_count", ...])
```

Node attributes added:

| Attribute | Source |
|---|---|
| `sent` | `sent_ctr` Counter |
| `received` | `recv_ctr` Counter |
| `months_active` | Length of each person's active-month set |
| `unique_contacts` | `G.degree(node)` — number of distinct email partners |

---

### Cell 5 — Centrality Metrics

**Purpose:** Computes five centrality measures. All heavy computations run on the **Largest Connected Component (LCC)** only; isolated sub-graphs are excluded.

| Metric | Algorithm | Notes |
|---|---|---|
| **Degree centrality** | `nx.degree_centrality` | Run on full graph `G`; fast, O(V) |
| **Weighted degree** | `G.degree(weight="weight")` | Sum of fractional edge weights |
| **Betweenness** | `nx.betweenness_centrality` (k=200) | Approximate using 200-node random sample for speed; still highly correlated with exact results at this scale |
| **PageRank** | `nx.pagerank` (α=0.85) | 100 iterations, tol=1e-4 |
| **Eigenvector** | `nx.eigenvector_centrality_numpy` | Falls back to a zero dict on convergence failure |
| **Clustering** | `nx.clustering` (unweighted) | Unweighted is 5–10× faster than weighted at this scale |
| **Closeness** | — | Skipped (O(V·E) is too slow at this graph size); stored as `0.0` |

---

### Cell 6 — Community Detection

**Purpose:** Assigns each node to a community using the **Louvain algorithm** (modularity maximisation) run on the LCC.

```python
partition = community_louvain.best_partition(Gcc, weight="weight", random_state=42)
```

Nodes not in the LCC receive `community = -1`. The number of detected communities is printed and stored in `stats.json`.

---

### Cell 7 — Role Inference

**Purpose:** Labels each node with a human-readable organisational role based on relative percentile thresholds, not fixed cutoffs.

| Role | Rule |
|---|---|
| **Executive / Broker** | PageRank ≥ 90th percentile **and** Betweenness ≥ 75th percentile |
| **Information Broker** | Betweenness ≥ 75th percentile only |
| **Broadcaster** | Sent > 500 emails and received < 30% of what they sent |
| **Information Sink** | Received > 3× what was sent and received > 200 emails |
| **Connector** | More than 50 unique contacts |
| **Regular Employee** | Falls into none of the above categories |

Thresholds are computed from the non-zero values of each metric array using `np.quantile`, making them resilient to graph size changes.

---

### Cell 8 — Temporal Analysis & Keywords

**Purpose:** Produces time-series email volume data and a word-frequency map from subject lines.

**Timeline:** Groups email counts by `YYYY-MM` period and sorts chronologically.

**Keywords:** Tokenises all subject lines with `re.findall(r"\b[a-z]{3,}\b", ...)`, removes a custom stop-word list (common English function words + email threading prefixes like `re`, `fw`, `fwd`), and returns the top 80 words by frequency.

**Degree distribution:** `Counter(d for _, d in G.degree())` — used to plot the log-log degree distribution in the dashboard.

---

### Cell 9 — JSON Export

**Purpose:** Serialises all analysis outputs to `sna_output/`.

**Graph data selection:** Only the top-500 nodes by `weighted_degree` are exported, and only edges where **both** endpoints are in that set. This keeps `graph_data.json` under ~5 MB while preserving the most-connected part of the network.

**Files written:**

| File | Contents | Format |
|---|---|---|
| `graph_data.json` | `{nodes: [...], edges: [...]}` — node objects with all metrics + edges | JSON (compact) |
| `top_nodes.json` | Top-30 nodes by PageRank with key metrics | JSON (compact) |
| `communities.json` | Community ID, size, top-5 members (by PageRank) | JSON (compact) |
| `timeline.json` | `[{period, email_count}, ...]` sorted by period | JSON (compact) |
| `keywords.json` | `[{word, count}, ...]` top-80 | JSON (compact) |
| `degree_dist.json` | `[{degree, count}, ...]` | JSON (compact) |
| `stats.json` | Graph-level statistics (pretty-printed) | JSON (indented) |
| `email_headers_metadata.csv` | One row per email: date, from, to_cc_count, subject | CSV |

**`stats.json` fields:**

| Field | Description |
|---|---|
| `total_emails` | Valid parsed emails |
| `total_nodes` | Nodes in the full graph |
| `total_edges` | Edges in the full graph |
| `n_communities` | Louvain communities detected |
| `density` | Graph density = 2E / (V(V-1)) |
| `avg_clustering` | Average clustering coefficient (LCC, unweighted) |
| `lcc_nodes` / `lcc_edges` | Size of the largest connected component |
| `date_range_start` / `date_range_end` | Earliest and latest email dates |

---

## Dashboard — `index.html`

The dashboard is a **self-contained single HTML file** (~3 300 lines) with no build step. It loads the generated JSON files via `fetch()` from `sna_output/` and renders a 3-D rotating sphere network using D3 v7.

### Header Bar

The header displays six live statistics loaded from `stats.json`:

| Stat | Description |
|---|---|
| **Emails** | Total parsed emails |
| **Nodes** | Total unique people (email addresses) |
| **Edges** | Total unique communication pairs |
| **Communities** | Louvain communities detected |
| **Density** | Graph density (4 decimal places) |
| **Date Range** | `YYYY-MM → YYYY-MM` covering the corpus |

The header also contains two **sidebar toggle buttons** (`<` / `>`) to collapse the left or right panel, giving more space to the network canvas. The **theme toggle** (☀/☾) switches between dark and light mode (preference is saved in `localStorage`).

---

### Left Sidebar — Controls

#### Graph Controls

| Control | Type | Effect |
|---|---|---|
| **Edge Weight Min** | Logarithmic slider | Hides edges below the selected weight threshold. Uses a log scale (`10^(v/30) - 1`) so fine-tuning near zero is easy. The displayed value updates in real time. |
| **Max Nodes** | Linear slider (10–500) | Limits how many nodes are drawn. Nodes are sorted by the current **Size By** metric before slicing, so the most-important nodes are always shown first. |
| **Sphere Radius** | Linear slider (15–90) | Controls the radius of the 3-D sphere as a percentage of `min(width, height)`. Larger values spread the nodes further apart. |
| **Spin Speed** | Linear slider (0–10) | Sets the axis-rotation speed of the auto-spin animation in radians per frame × 0.001. Zero = no spin. |

#### Display Toggles

| Toggle | Default | Effect |
|---|---|---|
| **Show Labels** | On | Toggles all node labels on/off. Also toggled by the `Ⓐ Labels` toolbar button and the `H` keyboard shortcut. |
| **Top Labels Only** | On | When on, only shows labels for nodes in the top 15th percentile of the current **Size By** metric. Reduces clutter with many nodes. |
| **Show Isolated** | Off | When off, nodes with no edges after the current weight filter are hidden. When on, isolated nodes are shown as floating points. |
| **Edge Width by Weight** | On | Scales edge stroke-width logarithmically with edge weight. When off, all edges are drawn at 0.6 px. |

#### View Buttons

| Button | Action |
|---|---|
| **▶ Spin** | Starts the auto-rotation animation (Y-axis). |
| **⏸ Pause** | Stops the auto-rotation. Dragging the canvas also pauses spin. |
| **↺ Reset** | Rebuilds the network from scratch (re-applies all current filter settings). |
| **⊡ Fit** | Resets the sphere rotation and re-centres the view. |
| **Unpin All** | Removes the "pinned" status from all nodes. |

#### Node Appearance

| Selector | Options | Effect |
|---|---|---|
| **Size By** | PageRank, Betweenness, Eigenvector, Weighted Degree, Degree Centrality | Controls which metric drives node radius (sqrt scale, range 3–20 px). |
| **Color By** | Community, Inferred Role, Betweenness, PageRank, Weighted Degree, Degree Centrality | Controls node fill colour. `Community` uses a fixed 18-colour categorical palette. `Inferred Role` uses the role-colour map. Continuous metrics use the D3 Plasma colour scale. |

#### Search

Type any part of a node ID or label in the search box. Up to 10 matching nodes appear in a dropdown; clicking one selects that node, pans the sphere to bring it into view, and opens the **Selected** panel on the right.

#### Legend

Switches dynamically based on **Color By**:

- **Community** — one coloured dot per community with member count; clicking a community dot selects all its visible nodes.
- **Inferred Role** — one entry per role type; clicking selects all visible nodes with that role.
- **Metric** — five percentile-range bands; clicking selects all nodes in that range.

The **Top 20 by PageRank** list below the legend always shows the 20 highest-PageRank nodes. Clicking a row selects and pans to that node.

---

### Main Canvas — Network Sphere

Nodes are distributed on the surface of a sphere using **Fibonacci-sphere latitude banding**: communities are assigned latitude bands so members of the same community cluster together visually. The Fibonacci golden-angle spacing prevents clustering within a band.

**Perspective projection** (`_ps = fov / (fov + rz)`) scales node radius by depth, giving a genuine 3-D feel. Nodes behind the sphere fade out; nodes at the front are brightest.

**Interactions:**

| Gesture | Effect |
|---|---|
| **Drag** (background or node) | Rotates the sphere via arcball rotation |
| **Scroll wheel** | Zooms in/out (adjusts sphere radius) |
| **Click** a node (Select mode) | Selects it and opens the detail panel |
| **Shift/Ctrl/Meta + Click** | Adds/removes a node from the multi-selection |
| **Alt + Click** _or_ Path tool + Click | Starts/completes a shortest-path query |
| **Double-click** a node | Pins/unpins the node (gold ring) and highlights its 1-hop ego network |

---

### Toolbar Buttons

| Button | Shortcut | Description |
|---|---|---|
| **☰ Select** | — | Default mode. Click to select, drag background to rotate sphere. |
| **⬡ Lasso** | `L` | Draw a rubber-band rectangle to select all nodes inside it. Shift+Lasso adds to the current selection. |
| **✥ Pan** | `P` | Switches to pan-only mode (drag moves the entire SVG viewport rather than rotating the sphere). |
| **Ⓐ Labels** | `H` | Toggles node label visibility. Synced with the left-panel toggle. |
| **⇄ Path** | Alt+Click | Activate then click two nodes to find and highlight the shortest path between them. The floating badge shows the hop count and node sequence. Click `[clear path]` or press `Esc` to dismiss. |
| **⬇ PNG** | — | Exports the current canvas as a 2× resolution PNG. Embedded styles are included so the download looks identical to the screen. |
| **{ } JSON** | — | Downloads the currently rendered nodes and edges as a formatted JSON file (`enron-sna-export.json`). |
| **? Help** | `?` | Opens the keyboard shortcuts overlay. |

---

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `L` | Toggle Lasso mode |
| `P` | Toggle Pan mode |
| `R` | Rebuild graph (reset filter + redraw) |
| `F` | Fit to view (reset rotation + centre) |
| `H` | Toggle node labels |
| `?` | Show / hide keyboard shortcuts overlay |
| `Esc` | Clear selection, exit current mode |
| `Shift + Click` | Multi-select nodes |
| `Double-click` | Pin or unpin a node |
| `Alt + Click` | Shortest-path start / end point |

---

### Right Sidebar — Analytics Panels

The right sidebar has three tabs:

#### Overview Tab (default)

| Widget | Description |
|---|---|
| **Community Sizes** (doughnut) | Relative sizes of the top-10 Louvain communities. Segments are coloured by the community palette. |
| **Top Communicators** (horizontal bar) | Top-10 nodes by PageRank showing Sent (cyan) and Received (red) email counts side by side. |
| **Subject Keywords** (word cloud) | Top-60 subject-line keywords scaled by frequency. Font size ranges from 9 px to 23 px; opacity from 0.35 to 1.0. |

#### Analytics Tab

| Widget | Description |
|---|---|
| **Email Volume Timeline** (area chart) | Monthly email count from the earliest to the latest date in the corpus. Useful for spotting the surge leading up to the Enron collapse (late 2001). |
| **Degree Distribution — log-log** (scatter) | Each point is `(log₁₀(degree), log₁₀(count))`. A straight line here indicates a power-law / scale-free degree distribution, typical of real-world communication networks. |
| **Community Stats** (table) | Community ID, member count, and top-3 members (by PageRank) for the 12 largest communities. |

#### Selected Tab

Appears when you click a node or make a lasso selection.

**Single-node detail:**

| Section | Information |
|---|---|
| Node name | Full email address |
| Role badge | Inferred organisational role (colour-coded) |
| Metric cards (2×4 grid) | PageRank, Betweenness, Eigenvector, Clustering, Sent, Received, Contacts, Months Active |
| Neighbours list | Top-15 neighbours by edge weight; clicking one navigates to that node |

**Multi-node detail (≥ 2 selected):**

Shows the count of selected nodes, aggregate stats (avg PageRank, avg Betweenness, total Sent, total Received), and two action buttons:

| Button | Action |
|---|---|
| **⊠ Only Selected** | Hides all other nodes and edges — isolates the selected subgraph. Click again (**⊡ Show All**) to restore the full view. |
| **✕ Clear** | Deselects all nodes and returns to normal view. |

---

### Tooltip

Hovering over any node shows a floating tooltip with:

- Full email address (monospace)
- Inferred role badge
- Community ID
- PageRank, Betweenness, Eigenvector, Clustering (5 decimal places)
- Sent / Received counts
- Unique contacts count
- Interaction hint

The tooltip repositions itself to stay within the canvas bounds.

---

## Output Files

### `graph_data.json`

```json
{
  "nodes": [
    {
      "id": "jeff.skilling@enron.com",
      "label": "jeff.skilling",
      "sent": 1402,
      "received": 3891,
      "unique_contacts": 312,
      "months_active": 24,
      "degree_centrality": 0.018372,
      "weighted_degree": 412.7,
      "betweenness": 0.004812,
      "closeness": 0.0,
      "eigenvector": 0.023145,
      "pagerank": 0.001823,
      "clustering": 0.3421,
      "community": 2,
      "inferred_role": "Executive / Broker"
    }
  ],
  "edges": [
    {
      "source": "jeff.skilling@enron.com",
      "target": "kenneth.lay@enron.com",
      "weight": 14.33,
      "raw_count": 42,
      "direct_count": 18,
      "broadcast_count": 9
    }
  ]
}
```

### `stats.json`

```json
{
  "total_emails": 517401,
  "total_nodes": 36692,
  "total_edges": 367662,
  "n_communities": 28,
  "density": 0.000546,
  "avg_clustering": 0.4812,
  "lcc_nodes": 33312,
  "lcc_edges": 365101,
  "date_range_start": "1999-11-01",
  "date_range_end": "2002-06-21"
}
```

---

## Algorithms & Design Decisions

### Tie-strength weighting (`1/N`)

A single email blast to 100 people should contribute less relationship strength per pair than a direct 1-to-1 message. Dividing the weight by recipient count `N` normalises for broadcast behaviour.

### Betweenness approximation (`k=200`)

Exact betweenness is O(VE) which is prohibitively slow for a graph with 30 K+ nodes. The `k=200` random-pivot approximation reduces this to O(k·E) with only ~5% error at this scale, and uses a fixed `seed=42` for reproducibility.

### Louvain on LCC only

Running community detection on the full graph would assign singleton communities to every isolated node. Restricting to the LCC produces meaningful communities reflecting the actual communication core of the organisation.

### Sphere layout (Fibonacci banding)

Rather than a force simulation (which would require running in the browser and prevents deterministic layout), nodes are placed on a sphere using Fibonacci golden-angle distribution within community latitude bands. This provides:
- Community grouping visible at a glance
- No computational cost at run time
- Stable layout on repeated loads

### `closeness = 0.0`

Closeness centrality requires a full BFS from every node — O(V·E) — which is too slow for this dataset. The field is preserved in the schema for forward compatibility but is always zero.

---

## Dependencies

### Python

| Package | Purpose |
|---|---|
| `pandas` | CSV chunked reading, DataFrame operations |
| `numpy` | Array quantile computations, float conversions |
| `networkx` | Graph construction, centrality algorithms, BFS |
| `python-louvain` | Louvain community detection (`community` module) |
| `tqdm` | Progress bars for the streaming loop |

Install with:

```bash
pip install pandas numpy networkx python-louvain tqdm
```

### Browser (CDN — no installation)

| Library | Version | Purpose |
|---|---|---|
| D3.js | 7.9.0 | Network rendering, force layout, zoom, scales |
| Chart.js | 4.4.1 | Analytics charts (doughnut, bar, line, scatter) |
| Google Fonts | — | Syne (UI) + Space Mono (monospace data) |

All assets are loaded from CDN; the dashboard works offline only if the CDN resources are cached.

---

## License

The Enron email dataset is public domain. Source code in this repository is provided for educational and research use.
