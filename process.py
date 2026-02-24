#!/usr/bin/env python
# coding: utf-8
"""
============================================================
ENRON EMAIL CORPUS — ADVANCED SOCIAL NETWORK ANALYSIS
============================================================
Author: Mayank
Purpose: Build a rich dataset for downstream power-level
         inference (via LLM) and real-vs-formal team
         visualization.

Key additions over v1:
  - Saves per-node email SAMPLES (up to N emails per person)
    so an LLM can later infer seniority / power level
    without reading every email.
  - Exports directed graph metrics (in-degree vs out-degree
    asymmetry is a strong seniority signal).
  - Exports community membership with bridging scores so
    a viz can contrast "real teams" vs org-chart teams.
  - All outputs are self-contained JSON / CSV — no extra
    state needed to continue to the visualization step.
============================================================
"""

import re
import json
import os
import gc
import warnings
from collections import Counter, defaultdict
from email.utils import parsedate
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMAIL_RE   = re.compile(r'[\w.+-]+@[\w.-]+\.[a-z]{2,}', re.I)
ENRON_RE   = re.compile(r'[\w.+-]+@[\w.-]*enron\.com',   re.I)

STOP_WORDS = {
    "re", "fw", "fwd", "the", "a", "an", "is", "in", "of", "to", "and",
    "for", "on", "at", "be", "with", "from", "your", "this", "that",
    "have", "will", "not", "are", "has", "was", "its", "it", "or",
    "but", "if", "as", "we", "you",
}

# How many representative email samples to keep per person for LLM analysis.
# Keeping this small (e.g. 10–20) means the LLM step later is cheap.
SAMPLES_PER_NODE = 15


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def parse_date_ymd(s: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (YYYY-MM-DD, YYYY-MM) from an email date string, or (None, None)."""
    try:
        t = parsedate(s)
        if not t:
            return None, None
        y = t[0]
        if y < 100:
            y += 2000 if y < 70 else 1900
        return f"{y:04d}-{t[1]:02d}-{t[2]:02d}", f"{y:04d}-{t[1]:02d}"
    except Exception:
        return None, None


def extract_entities(row) -> Optional[Tuple[str, tuple, str, str, str]]:
    """
    Validate a CSV row and extract:
      (sender, enron_recipients, date_str, subject, body_snippet)
    Returns None if the row is unusable.
    """
    from_str = str(row.From)
    if from_str == "nan":
        return None
    senders = EMAIL_RE.findall(from_str)
    if not senders:
        return None
    sender = senders[0].lower()
    if not ENRON_RE.search(sender):
        return None

    to_str = str(row.To)
    if to_str == "nan":
        return None
    all_recips = EMAIL_RE.findall(to_str)
    enron_recips = tuple(
        e.lower() for e in all_recips
        if e.lower() != sender and ENRON_RE.search(e)
    )
    if not enron_recips:
        return None

    date_str    = str(row.Date)    if str(row.Date)    != "nan" else ""
    subject     = str(row.Subject) if str(row.Subject) != "nan" else ""

    # Body column is optional — gracefully degrade if absent
    try:
        body = str(row.body) if str(row.body) != "nan" else ""
    except AttributeError:
        body = ""

    # Keep only first 300 chars of body — enough context for an LLM
    body_snippet = body[:300].strip()

    return sender, enron_recips, date_str, subject, body_snippet


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class EnronNetworkAnalyzer:
    """
    End-to-end pipeline:
      1. stream_and_parse   — read CSV, build edge lists + node samples
      2. aggregate_edges    — collapse to undirected weighted edges
      3. build_graph        — construct NetworkX graph
      4. compute_metrics    — centralities, communities
      5. infer_roles        — heuristic role labels (augmented by LLM later)
      6. export_results     — write all JSON / CSV outputs
    """

    def __init__(self, input_csv: str, output_dir: str, chunk_size: int = 25_000):
        self.input_csv  = input_csv
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)

        # ── Edge accumulators ──────────────────────────────────────────────
        self.src_list = []
        self.tgt_list = []
        self.w_list   = []
        self.rc_list  = []
        self.dc_list  = []
        self.bc_list  = []

        # ── Node-level counters ────────────────────────────────────────────
        self.sent_ctr   = Counter()   # emails sent
        self.recv_ctr   = Counter()   # emails received
        self.month_sets = defaultdict(set)

        # ── LLM power-level sample bank ───────────────────────────────────
        # { node_email: [ {date, to, subject, body_snippet}, ... ] }
        # We keep a STRATIFIED sample: earliest + latest + high-recipient emails.
        # Simple reservoir sampling is used here for memory efficiency.
        self._sample_bank: dict[str, list] = defaultdict(list)
        self._sample_counts: Counter = Counter()   # tracks how many we've seen

        # ── Timeline / keyword raw data ───────────────────────────────────
        self.subj_list  = []
        self.dates_list = []
        self.ym_list    = []
        self.headers_meta = []

        # ── Totals ────────────────────────────────────────────────────────
        self.total_raw = 0
        self.total_ok  = 0

        # ── Graph objects (populated in later stages) ─────────────────────
        self.df_agg      = None
        self.G           = None   # full undirected graph
        self.G_directed  = None   # directed graph for asymmetry signals
        self.Gcc         = None   # largest connected component
        self.n_communities = 0

    # ------------------------------------------------------------------
    # Stage 1: Streaming parse
    # ------------------------------------------------------------------

    def _reservoir_sample(self, node: str, record: dict):
        """
        Keep up to SAMPLES_PER_NODE records per node via reservoir sampling
        so we get a representative spread without blowing up memory.
        """
        self._sample_counts[node] += 1
        k = self._sample_counts[node]
        bank = self._sample_bank[node]

        if len(bank) < SAMPLES_PER_NODE:
            bank.append(record)
        else:
            # Replace a random existing entry with decreasing probability
            j = int(np.random.randint(0, k))
            if j < SAMPLES_PER_NODE:
                bank[j] = record

    def stream_and_parse(self):
        """Read CSV in chunks, build edge lists and per-node email samples."""
        print("⚙️  Streaming CSV …")

        # Try to detect whether a 'body' column exists
        probe = pd.read_csv(self.input_csv, nrows=1, dtype=str, engine="c")
        has_body = "body" in [c.lower() for c in probe.columns]
        cols = ["Date", "From", "To", "Subject"] + (["body"] if has_body else [])

        reader = pd.read_csv(
            self.input_csv,
            usecols=cols,
            chunksize=self.chunk_size,
            dtype=str,
            engine="c",
            on_bad_lines="skip",
        )

        for chunk in tqdm(reader, desc="  Chunks"):
            self.total_raw += len(chunk)

            for row in chunk.itertuples(index=False):
                result = extract_entities(row)
                if result is None:
                    continue

                sender, enron_recips, date_str, subject, body_snippet = result
                date, ym = parse_date_ymd(date_str)
                n_r = len(enron_recips)
                w   = 1.0 / n_r

                # ── Node counters ──────────────────────────────────────
                self.sent_ctr[sender] += 1
                if ym:
                    self.month_sets[sender].add(ym)

                self.subj_list.append(subject)
                self.dates_list.append(date)
                self.ym_list.append(ym)
                self.headers_meta.append({
                    "date":         date or "",
                    "from":         sender,
                    "to_cc_count":  n_r,
                    "subject":      subject,
                })

                # ── LLM sample bank ────────────────────────────────────
                sample_record = {
                    "date":         date or "",
                    "to":           list(enron_recips)[:5],   # cap list length
                    "subject":      subject,
                    "body_snippet": body_snippet,
                    "recipient_count": n_r,
                }
                self._reservoir_sample(sender, sample_record)

                # ── Edge lists ─────────────────────────────────────────
                for r in enron_recips:
                    self.src_list.append(sender)
                    self.tgt_list.append(r)
                    self.w_list.append(w)
                    self.rc_list.append(1)
                    self.dc_list.append(1 if n_r == 1 else 0)
                    self.bc_list.append(1 if n_r > 5 else 0)

                    self.recv_ctr[r] += 1
                    if ym:
                        self.month_sets[r].add(ym)

                self.total_ok += 1

            gc.collect()

        print(f"   {self.total_ok:,} valid emails from {self.total_raw:,} rows")

    # ------------------------------------------------------------------
    # Stage 2: Edge aggregation
    # ------------------------------------------------------------------

    def aggregate_edges(self):
        """Vectorised collapse of raw edges into weighted undirected edges."""
        print("🔗 Aggregating edges …")

        df_e = pd.DataFrame({
            "src": pd.array(self.src_list, dtype="string"),
            "tgt": pd.array(self.tgt_list, dtype="string"),
            "w":   np.array(self.w_list,   dtype="float32"),
            "rc":  np.array(self.rc_list,  dtype="int32"),
            "dc":  np.array(self.dc_list,  dtype="int32"),
            "bc":  np.array(self.bc_list,  dtype="int32"),
        })

        # Also build DIRECTED graph data before canonicalising
        self._directed_edges = df_e[["src", "tgt", "rc"]].copy()

        # Free raw lists
        self.src_list, self.tgt_list, self.w_list = [], [], []
        self.rc_list,  self.dc_list,  self.bc_list = [], [], []
        gc.collect()

        # Canonicalise (smaller string → src) for undirected aggregation
        swap = df_e["src"] > df_e["tgt"]
        df_e.loc[swap, ["src", "tgt"]] = df_e.loc[swap, ["tgt", "src"]].values

        self.df_agg = (
            df_e.groupby(["src", "tgt"], sort=False)
            .agg(
                weight         =("w",  "sum"),
                raw_count      =("rc", "sum"),
                direct_count   =("dc", "sum"),
                broadcast_count=("bc", "sum"),
            )
            .reset_index()
        )

        del df_e
        gc.collect()
        print(f"   {len(self.df_agg):,} unique undirected edges")

    # ------------------------------------------------------------------
    # Stage 3: Graph construction
    # ------------------------------------------------------------------

    def build_graph(self):
        """Build undirected + directed NetworkX graphs."""
        print("📊 Building graphs …")

        # Undirected (for community detection & most centralities)
        self.G = nx.from_pandas_edgelist(
            self.df_agg, source="src", target="tgt",
            edge_attr=["weight", "raw_count", "direct_count", "broadcast_count"],
        )

        # Directed (for in/out asymmetry — key power signal)
        dir_agg = (
            self._directed_edges
            .groupby(["src", "tgt"], sort=False)["rc"]
            .sum()
            .reset_index()
            .rename(columns={"rc": "weight"})
        )
        self.G_directed = nx.from_pandas_edgelist(
            dir_agg, source="src", target="tgt",
            edge_attr=["weight"],
            create_using=nx.DiGraph(),
        )
        del self._directed_edges, dir_agg, self.df_agg
        gc.collect()

        # Populate base node attributes
        months_active = {n: len(s) for n, s in self.month_sets.items()}
        self.month_sets.clear()
        gc.collect()

        for node in self.G.nodes():
            self.G.nodes[node].update({
                "sent":            self.sent_ctr.get(node, 0),
                "received":        self.recv_ctr.get(node, 0),
                "months_active":   months_active.get(node, 0),
                "unique_contacts": self.G.degree(node),
                # Directed asymmetry: positive → sends more than receives
                "out_in_ratio":    round(
                    self.sent_ctr.get(node, 0) /
                    max(self.recv_ctr.get(node, 1), 1), 3
                ),
            })

        print(f"   {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")

    # ------------------------------------------------------------------
    # Stage 4: Centrality & community metrics
    # ------------------------------------------------------------------

    def compute_metrics(self):
        """Centralities and Louvain communities on the LCC."""
        print("📐 Computing centralities & communities …")

        lcc_nodes = max(nx.connected_components(self.G), key=len)
        self.Gcc  = self.G.subgraph(lcc_nodes).copy()

        degree_cent  = nx.degree_centrality(self.G)
        weighted_deg = dict(self.G.degree(weight="weight"))
        clustering   = nx.clustering(self.G)

        k_approx    = min(200, self.Gcc.number_of_nodes())
        betweenness = nx.betweenness_centrality(
            self.Gcc, weight="weight", normalized=True, k=k_approx, seed=42
        )
        pagerank = nx.pagerank(
            self.Gcc, weight="weight", alpha=0.85, max_iter=100, tol=1e-4
        )

        try:
            eigenvector = nx.eigenvector_centrality_numpy(self.Gcc, weight="weight")
        except Exception:
            eigenvector = {}

        # ── Community detection ────────────────────────────────────────
        partition = community_louvain.best_partition(
            self.Gcc, weight="weight", random_state=42
        )
        self.n_communities = len(set(partition.values()))

        # ── Bridging score: how many distinct communities does a node connect?
        # High bridging → cross-team connector ("information broker")
        bridge_score: dict[str, int] = defaultdict(int)
        for u, v in self.Gcc.edges():
            if partition.get(u, -1) != partition.get(v, -1):
                bridge_score[u] += 1
                bridge_score[v] += 1

        # ── Map metrics back to main graph ────────────────────────────
        for node in self.G.nodes():
            self.G.nodes[node].update({
                "degree_centrality": round(degree_cent.get(node, 0),    6),
                "weighted_degree":   round(float(weighted_deg.get(node, 0)), 3),
                "betweenness":       round(betweenness.get(node, 0),    6),
                "pagerank":          round(pagerank.get(node, 0),       6),
                "eigenvector":       round(eigenvector.get(node, 0),    6),
                "clustering":        round(clustering.get(node, 0),     5),
                "community":         partition.get(node, -1),
                "bridge_score":      bridge_score.get(node, 0),
            })

    # ------------------------------------------------------------------
    # Stage 5: Heuristic role inference
    # ------------------------------------------------------------------

    def infer_roles(self):
        """
        Assign a preliminary role label using structural + behavioral signals.
        This is intentionally a rough classification — the LLM step will refine
        'power level' in a later pass using the saved email samples.

        Role labels are designed to map onto a power hierarchy:
          Executive / Broker  →  highest
          Information Broker  →  high
          Broadcaster         →  medium-high (e.g. announcement bots)
          Connector           →  medium
          Information Sink    →  medium-low (passive receivers)
          Regular Employee    →  baseline
        """
        print("🎭 Inferring heuristic roles …")

        pr_arr  = np.array([self.G.nodes[n]["pagerank"]    for n in self.G.nodes()])
        bet_arr = np.array([self.G.nodes[n]["betweenness"] for n in self.G.nodes()])

        pr_hi  = float(np.quantile(pr_arr [pr_arr  > 0], 0.90)) if pr_arr.any()  else 0.01
        bet_hi = float(np.quantile(bet_arr[bet_arr > 0], 0.75)) if bet_arr.any() else 0.01

        for node in self.G.nodes():
            d    = self.G.nodes[node]
            pr   = d["pagerank"]
            bet  = d["betweenness"]
            sent = d.get("sent", 0)
            recv = d.get("received", 0)
            uc   = d.get("unique_contacts", 0)

            if pr >= pr_hi and bet >= bet_hi:
                role = "Executive / Broker"
            elif bet >= bet_hi:
                role = "Information Broker"
            elif sent > 500 and recv < sent * 0.3:
                role = "Broadcaster"
            elif recv > sent * 3 and recv > 200:
                role = "Information Sink"
            elif uc > 50:
                role = "Connector"
            else:
                role = "Regular Employee"

            self.G.nodes[node]["inferred_role"] = role

        # Assign a numeric power_level_heuristic (0–5) for the viz
        _role_rank = {
            "Executive / Broker": 5,
            "Information Broker": 4,
            "Broadcaster":        3,
            "Connector":          3,
            "Information Sink":   2,
            "Regular Employee":   1,
        }
        for node in self.G.nodes():
            role = self.G.nodes[node]["inferred_role"]
            self.G.nodes[node]["power_level_heuristic"] = _role_rank.get(role, 1)

    # ------------------------------------------------------------------
    # Stage 6: Export
    # ------------------------------------------------------------------

    def export_results(self):
        """Write every dataset needed by the downstream visualization + LLM steps."""
        print("💾 Exporting …")

        sep = (",", ":")

        # ── Timeline ──────────────────────────────────────────────────
        ym_ctr = Counter(ym for ym in self.ym_list if ym)
        timeline_data = sorted(
            [{"period": k, "email_count": v} for k, v in ym_ctr.items()],
            key=lambda x: x["period"],
        )

        # ── Keywords ──────────────────────────────────────────────────
        wc = Counter()
        for s in self.subj_list:
            for w in re.findall(r"\b[a-z]{3,}\b", s.lower()):
                if w not in STOP_WORDS:
                    wc[w] += 1
        top_keywords = [{"word": w, "count": c} for w, c in wc.most_common(80)]

        # ── Degree distribution ───────────────────────────────────────
        degree_dist      = Counter(d for _, d in self.G.degree())
        degree_dist_data = [{"degree": k, "count": v} for k, v in sorted(degree_dist.items())]

        # ── Node selection for graph export ───────────────────────────
        # Export top-500 by weighted_degree so the viz stays responsive
        top_node_ids = sorted(
            self.G.nodes(),
            key=lambda n: self.G.nodes[n].get("weighted_degree", 0),
            reverse=True,
        )[:500]
        top_node_set = set(top_node_ids)

        nodes_out = []
        for n in top_node_ids:
            d = self.G.nodes[n]
            nodes_out.append({
                "id":                    n,
                "label":                 n.split("@")[0],
                "sent":                  d.get("sent", 0),
                "received":              d.get("received", 0),
                "unique_contacts":       d.get("unique_contacts", 0),
                "months_active":         d.get("months_active", 0),
                "out_in_ratio":          d.get("out_in_ratio", 1.0),
                "degree_centrality":     d.get("degree_centrality", 0),
                "weighted_degree":       d.get("weighted_degree", 0),
                "betweenness":           d.get("betweenness", 0),
                "eigenvector":           d.get("eigenvector", 0),
                "pagerank":              d.get("pagerank", 0),
                "clustering":            d.get("clustering", 0),
                "community":             d.get("community", -1),
                "bridge_score":          d.get("bridge_score", 0),
                "inferred_role":         d.get("inferred_role", "Unknown"),
                "power_level_heuristic": d.get("power_level_heuristic", 1),
                # Placeholder for LLM-assigned values (filled in later)
                "power_level_llm":       None,
                "llm_reasoning":         "",
            })

        edges_out = [
            {
                "source":          u,
                "target":          v,
                "weight":          round(float(d.get("weight", 0)), 3),
                "raw_count":       int(d.get("raw_count", 0)),
                "direct_count":    int(d.get("direct_count", 0)),
                "broadcast_count": int(d.get("broadcast_count", 0)),
                # Is this a cross-community edge? (key for "real teams" viz)
                "cross_community": (
                    self.G.nodes[u].get("community", -1) !=
                    self.G.nodes[v].get("community", -2)
                ),
            }
            for u, v, d in self.G.edges(data=True)
            if u in top_node_set and v in top_node_set
        ]

        # ── Top nodes summary ─────────────────────────────────────────
        top_nodes_data = [
            {
                "id":               n,
                "sent":             d.get("sent", 0),
                "received":         d.get("received", 0),
                "unique_contacts":  d.get("unique_contacts", 0),
                "out_in_ratio":     d.get("out_in_ratio", 1.0),
                "pagerank":         round(d.get("pagerank", 0), 6),
                "betweenness":      round(d.get("betweenness", 0), 6),
                "bridge_score":     d.get("bridge_score", 0),
                "community":        d.get("community", -1),
                "inferred_role":    d.get("inferred_role", "Unknown"),
                "power_level_heuristic": d.get("power_level_heuristic", 1),
                "power_level_llm":  None,
                "llm_reasoning":    "",
            }
            for n, d in sorted(
                self.G.nodes(data=True),
                key=lambda x: x[1].get("pagerank", 0),
                reverse=True,
            )[:50]   # top-50 for the summary panel
        ]

        # ── Communities summary ───────────────────────────────────────
        comm_members: dict[int, list] = defaultdict(list)
        for n in top_node_ids:
            comm_members[self.G.nodes[n].get("community", -1)].append(n)

        community_summary = []
        for cid, ms in sorted(comm_members.items(), key=lambda x: -len(x[1])):
            top_members = sorted(
                ms, key=lambda n: self.G.nodes[n].get("pagerank", 0), reverse=True
            )[:5]
            # Average heuristic power level of community
            avg_power = round(
                np.mean([self.G.nodes[n].get("power_level_heuristic", 1) for n in ms]), 2
            )
            community_summary.append({
                "community_id": cid,
                "size":         len(ms),
                "avg_power_level_heuristic": avg_power,
                "top_members":  top_members,
                # Cross-community edges FROM this community
                "external_edges": sum(
                    1 for n in ms
                    for nb in self.G.neighbors(n)
                    if self.G.nodes.get(nb, {}).get("community", -2) != cid
                ),
            })

        # ── LLM sample bank export ────────────────────────────────────
        # Saved for ALL nodes in the top-500, not just the top-30,
        # so the power-level inference step has broad coverage.
        llm_samples = {
            node: self._sample_bank.get(node, [])
            for node in top_node_ids
        }

        # ── Global stats ──────────────────────────────────────────────
        valid_dates = [d for d in self.dates_list if d]
        stats = {
            "total_emails":    self.total_ok,
            "total_nodes":     self.G.number_of_nodes(),
            "total_edges":     self.G.number_of_edges(),
            "n_communities":   self.n_communities,
            "density":         round(nx.density(self.G), 6),
            "avg_clustering":  round(nx.average_clustering(self.Gcc), 4) if self.Gcc else 0,
            "lcc_nodes":       self.Gcc.number_of_nodes() if self.Gcc else 0,
            "lcc_edges":       self.Gcc.number_of_edges() if self.Gcc else 0,
            "date_range_start": min(valid_dates, default=""),
            "date_range_end":   max(valid_dates, default=""),
            "samples_per_node": SAMPLES_PER_NODE,
        }

        # ── Write files ───────────────────────────────────────────────
        out = self.output_dir

        pd.DataFrame(
            self.headers_meta,
            columns=["date", "from", "to_cc_count", "subject"],
        ).to_csv(f"{out}/email_headers_metadata.csv", index=False)

        with open(f"{out}/graph_data.json",     "w") as f:
            json.dump({"nodes": nodes_out, "edges": edges_out}, f, separators=sep)

        with open(f"{out}/timeline.json",        "w") as f:
            json.dump(timeline_data,    f, separators=sep)

        with open(f"{out}/keywords.json",        "w") as f:
            json.dump(top_keywords,     f, separators=sep)

        with open(f"{out}/degree_dist.json",     "w") as f:
            json.dump(degree_dist_data, f, separators=sep)

        with open(f"{out}/top_nodes.json",       "w") as f:
            json.dump(top_nodes_data,   f, separators=sep)

        with open(f"{out}/communities.json",     "w") as f:
            json.dump(community_summary, f, separators=sep)

        with open(f"{out}/stats.json",           "w") as f:
            json.dump(stats, f, indent=2)

        # ── LLM inputs (the key new output) ──────────────────────────
        # llm_power_inputs.json  — feed this to your LLM in the next step.
        # Format:
        #   [ { "email": "...", "sample_emails": [...] }, ... ]
        # Each entry is one person; the LLM reads their sample emails and
        # assigns a power_level (1–10) + short reasoning string.
        llm_input_list = [
            {
                "email":         node,
                "label":         node.split("@")[0],
                "heuristic_role": self.G.nodes[node].get("inferred_role", "Unknown"),
                "power_level_heuristic": self.G.nodes[node].get("power_level_heuristic", 1),
                "stats": {
                    "sent":           self.G.nodes[node].get("sent", 0),
                    "received":       self.G.nodes[node].get("received", 0),
                    "unique_contacts": self.G.nodes[node].get("unique_contacts", 0),
                    "out_in_ratio":   self.G.nodes[node].get("out_in_ratio", 1.0),
                    "pagerank":       self.G.nodes[node].get("pagerank", 0),
                    "betweenness":    self.G.nodes[node].get("betweenness", 0),
                },
                "sample_emails": llm_samples.get(node, []),
            }
            for node in top_node_ids
        ]

        with open(f"{out}/llm_power_inputs.json", "w") as f:
            json.dump(llm_input_list, f, separators=sep)

        print(f"\n✅  Done! Files in: {out}/")
        print()
        _col_w = max(len(k) for k in stats) + 2
        for k, v in stats.items():
            print(f"   {k:<{_col_w}} {v}")

        print("\n📁  Output files:")
        for fname in sorted(os.listdir(out)):
            size = os.path.getsize(f"{out}/{fname}")
            print(f"   {fname:<35} {size/1024:>8.1f} KB")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full pipeline."""
        self.stream_and_parse()
        self.aggregate_edges()
        self.build_graph()
        self.compute_metrics()
        self.infer_roles()
        self.export_results()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT_FILE    = "enron_internal_only.csv"
    OUTPUT_FOLDER = "sna_output"

    if os.path.exists(INPUT_FILE):
        analyzer = EnronNetworkAnalyzer(INPUT_FILE, OUTPUT_FOLDER)
        analyzer.run()
    else:
        print(f"❌  {INPUT_FILE} not found. Run the filtering script first.")