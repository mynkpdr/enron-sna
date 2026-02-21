# ============================================================
# ENRON EMAIL CORPUS — ADVANCED SOCIAL NETWORK ANALYSIS
# Google Colab Script  ·  OPTIMISED FOR SPEED & LOW RAM
# ============================================================
# Key optimisations vs. original:
#   • CSV streamed in chunks → never loads 1.3 GB into RAM
#   • Header-only regex parser (no email.message_from_string)
#   • Vectorised pandas groupby instead of iterrows
#   • Edge accumulation with flat lists → one-shot DataFrame
#   • Betweenness k=200 sample (fast approx, still accurate)
#   • closeness skipped (O(V·E), too slow at this scale)
#   • avg_clustering on LCC only (not whole graph)
#   • Louvain on LCC only
#   • gc.collect() after each heavy step
# ============================================================

import subprocess, sys


def pip(*a):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *a])


pip("pandas", "numpy", "networkx", "python-louvain", "tqdm", "pyarrow")

import re, json, os, gc, warnings
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from email.utils import parsedate

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
EMAILS_CSV = "emails.csv"
OUTPUT_DIR = "sna_output"
CHUNK_SIZE = 20_000  # rows per chunk — lower if still OOM
ENRON_RE = re.compile(r"[\w.+-]+@[\w.-]*enron\.com", re.I)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FAST HEADER PARSER
# Scans only first 2000 chars with regex — ~15x faster than
# email.message_from_string which also parses the full body.
# ─────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
_FROM_RE = re.compile(r"^From\s*:\s*(.+)", re.I | re.M)
_TO_RE = re.compile(r"^To\s*:\s*(.*?)(?=\n\S|\n\n|\Z)", re.I | re.M | re.S)
_CC_RE = re.compile(r"^(?:Cc|Bcc)\s*:\s*(.*?)(?=\n\S|\n\n|\Z)", re.I | re.M | re.S)
_DATE_RE = re.compile(r"^Date\s*:\s*(.+)", re.I | re.M)
_SUBJ_RE = re.compile(r"^Subject\s*:\s*(.+)", re.I | re.M)


def fast_parse(raw: str):
    """Return (sender, enron_recips_tuple, date_str, subject) or None."""
    head = raw[:2000]

    m = _FROM_RE.search(head)
    if not m:
        return None
    emails_in_from = _EMAIL_RE.findall(m.group(1))
    if not emails_in_from:
        return None
    sender = emails_in_from[0].lower()
    if not ENRON_RE.search(sender):
        return None

    to_txt = (_TO_RE.search(head) or type("", (), {"group": lambda s, i: ""})()).group(
        1
    ) or ""
    cc_txt = (_CC_RE.search(head) or type("", (), {"group": lambda s, i: ""})()).group(
        1
    ) or ""
    all_recips = _EMAIL_RE.findall(to_txt + " " + cc_txt)
    enron_recips = tuple(
        e.lower() for e in all_recips if e.lower() != sender and ENRON_RE.search(e)
    )
    if not enron_recips:
        return None

    date_m = _DATE_RE.search(head)
    date_str = date_m.group(1).strip() if date_m else ""
    subj_m = _SUBJ_RE.search(head)
    subject = subj_m.group(1).strip() if subj_m else ""

    return sender, enron_recips, date_str, subject


def parse_date_ymd(s: str):
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


# ─────────────────────────────────────────────────────────────
# STREAMING PASS — chunk the CSV, parse, accumulate edges
# ─────────────────────────────────────────────────────────────
print("⚙️  Streaming CSV …")

src_list = []
tgt_list = []
w_list = []
rc_list = []
dc_list = []
bc_list = []

sent_ctr = Counter()
recv_ctr = Counter()
month_sets = defaultdict(set)
subj_list = []
dates_list = []
ym_list = []

total_raw = total_ok = 0

for chunk in tqdm(
    pd.read_csv(
        EMAILS_CSV,
        usecols=["message"],
        chunksize=CHUNK_SIZE,
        dtype=str,
        engine="c",
        on_bad_lines="skip",
    ),
    desc="  chunks",
):
    total_raw += len(chunk)
    for raw in chunk["message"]:
        if not isinstance(raw, str):
            continue
        result = fast_parse(raw)
        if result is None:
            continue
        sender, enron_recips, date_str, subject = result

        date, ym = parse_date_ymd(date_str)
        n_r = len(enron_recips)
        w = 1.0 / n_r

        sent_ctr[sender] += 1
        if ym:
            month_sets[sender].add(ym)
        subj_list.append(subject)
        dates_list.append(date)
        ym_list.append(ym)

        for r in enron_recips:
            src_list.append(sender)
            tgt_list.append(r)
            w_list.append(w)
            rc_list.append(1)
            dc_list.append(1 if n_r == 1 else 0)
            bc_list.append(1 if n_r > 5 else 0)
            recv_ctr[r] += 1
            if ym:
                month_sets[r].add(ym)

        total_ok += 1

    gc.collect()

print(f"   {total_ok:,} valid emails parsed from {total_raw:,} rows")
print(f"   {len(src_list):,} directed edge occurrences collected")

# ─────────────────────────────────────────────────────────────
# AGGREGATE EDGES — vectorised groupby
# ─────────────────────────────────────────────────────────────
print("🔗 Aggregating edges …")

df_e = pd.DataFrame(
    {
        "src": pd.array(src_list, dtype="string"),
        "tgt": pd.array(tgt_list, dtype="string"),
        "w": np.array(w_list, dtype="float32"),
        "rc": np.array(rc_list, dtype="int32"),
        "dc": np.array(dc_list, dtype="int32"),
        "bc": np.array(bc_list, dtype="int32"),
    }
)
del src_list, tgt_list, w_list, rc_list, dc_list, bc_list
gc.collect()

# Canonicalise to undirected
swap = df_e["src"] > df_e["tgt"]
df_e.loc[swap, ["src", "tgt"]] = df_e.loc[swap, ["tgt", "src"]].values

df_agg = (
    df_e.groupby(["src", "tgt"], sort=False)
    .agg(
        weight=("w", "sum"),
        raw_count=("rc", "sum"),
        direct_count=("dc", "sum"),
        broadcast_count=("bc", "sum"),
    )
    .reset_index()
)
del df_e
gc.collect()
print(f"   {len(df_agg):,} undirected edges")

# ─────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────
print("📊 Building graph …")
G = nx.from_pandas_edgelist(
    df_agg,
    source="src",
    target="tgt",
    edge_attr=["weight", "raw_count", "direct_count", "broadcast_count"],
)
del df_agg
gc.collect()

months_active = {n: len(s) for n, s in month_sets.items()}
del month_sets
gc.collect()

for node in G.nodes():
    G.nodes[node]["sent"] = sent_ctr.get(node, 0)
    G.nodes[node]["received"] = recv_ctr.get(node, 0)
    G.nodes[node]["months_active"] = months_active.get(node, 0)
    G.nodes[node]["unique_contacts"] = G.degree(node)

print(f"   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ─────────────────────────────────────────────────────────────
# CENTRALITY — on LCC, fast settings
# ─────────────────────────────────────────────────────────────
print("📐 Centrality metrics …")
lcc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(lcc_nodes).copy()
print(f"   LCC: {Gcc.number_of_nodes()} nodes, {Gcc.number_of_edges()} edges")

degree_cent = nx.degree_centrality(G)
weighted_deg = dict(G.degree(weight="weight"))

K = min(200, Gcc.number_of_nodes())
print(f"   Betweenness (k={K} approx) …")
betweenness = nx.betweenness_centrality(
    Gcc, weight="weight", normalized=True, k=K, seed=42
)

print("   PageRank …")
pagerank = nx.pagerank(Gcc, weight="weight", alpha=0.85, max_iter=100, tol=1e-4)

print("   Eigenvector …")
try:
    eigenvector = nx.eigenvector_centrality_numpy(Gcc, weight="weight")
except Exception:
    eigenvector = {}

print("   Clustering (unweighted, fast) …")
clustering = nx.clustering(G)  # unweighted is 5-10x faster

# closeness = skipped at this scale (O(V·E) is too slow)
for node in G.nodes():
    G.nodes[node]["degree_centrality"] = round(degree_cent.get(node, 0), 6)
    G.nodes[node]["weighted_degree"] = round(float(weighted_deg.get(node, 0)), 3)
    G.nodes[node]["betweenness"] = round(betweenness.get(node, 0), 6)
    G.nodes[node]["pagerank"] = round(pagerank.get(node, 0), 6)
    G.nodes[node]["eigenvector"] = round(eigenvector.get(node, 0), 6)
    G.nodes[node]["closeness"] = 0.0
    G.nodes[node]["clustering"] = round(clustering.get(node, 0), 5)

# ─────────────────────────────────────────────────────────────
# COMMUNITY DETECTION
# ─────────────────────────────────────────────────────────────
print("🏘️  Louvain communities …")
partition = community_louvain.best_partition(Gcc, weight="weight", random_state=42)
for node in G.nodes():
    G.nodes[node]["community"] = partition.get(node, -1)
n_communities = len(set(partition.values()))
print(f"   {n_communities} communities")

# ─────────────────────────────────────────────────────────────
# ROLE INFERENCE — percentile-relative thresholds
# ─────────────────────────────────────────────────────────────
print("🎭 Role inference …")
pr_arr = np.array([G.nodes[n]["pagerank"] for n in G.nodes()])
bet_arr = np.array([G.nodes[n]["betweenness"] for n in G.nodes()])
pr_hi = float(np.quantile(pr_arr[pr_arr > 0], 0.90)) if pr_arr.any() else 0.01
bet_hi = float(np.quantile(bet_arr[bet_arr > 0], 0.75)) if bet_arr.any() else 0.01

for node in G.nodes():
    d = G.nodes[node]
    pr = d["pagerank"]
    bet = d["betweenness"]
    sent = d.get("sent", 0)
    recv = d.get("received", 0)
    uc = d.get("unique_contacts", 0)
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
    G.nodes[node]["inferred_role"] = role

# ─────────────────────────────────────────────────────────────
# TEMPORAL, KEYWORDS, DEGREE DIST
# ─────────────────────────────────────────────────────────────
print("📅 Temporal / keywords …")
ym_ctr = Counter(ym for ym in ym_list if ym)
timeline_data = sorted(
    [{"period": k, "email_count": v} for k, v in ym_ctr.items()],
    key=lambda x: x["period"],
)
del ym_list
gc.collect()

STOP = {
    "re",
    "fw",
    "fwd",
    "the",
    "a",
    "an",
    "is",
    "in",
    "of",
    "to",
    "and",
    "for",
    "on",
    "at",
    "be",
    "with",
    "from",
    "your",
    "this",
    "that",
    "have",
    "will",
    "not",
    "are",
    "has",
    "was",
    "its",
    "it",
    "or",
    "but",
    "if",
    "as",
    "we",
    "you",
}
wc = Counter()
for s in subj_list:
    for w in re.findall(r"\b[a-z]{3,}\b", s.lower()):
        if w not in STOP:
            wc[w] += 1
top_keywords = [{"word": w, "count": c} for w, c in wc.most_common(80)]
del subj_list
gc.collect()

degree_dist = Counter(d for _, d in G.degree())
degree_dist_data = [{"degree": k, "count": v} for k, v in sorted(degree_dist.items())]

# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────
print("💾 Exporting JSON …")
top_node_ids = sorted(
    G.nodes(), key=lambda n: G.nodes[n].get("weighted_degree", 0), reverse=True
)[:500]
top_node_set = set(top_node_ids)

nodes_out = [
    {
        k: G.nodes[n].get(k, 0 if k not in ("id", "label", "inferred_role") else "")
        for k in (
            "id",
            "label",
            "sent",
            "received",
            "unique_contacts",
            "months_active",
            "degree_centrality",
            "weighted_degree",
            "betweenness",
            "closeness",
            "eigenvector",
            "pagerank",
            "clustering",
            "community",
            "inferred_role",
        )
    }
    | {"id": n, "label": n.split("@")[0]}
    for n in top_node_ids
]

edges_out = [
    {
        "source": u,
        "target": v,
        "weight": round(float(d.get("weight", 0)), 3),
        "raw_count": int(d.get("raw_count", 0)),
        "direct_count": int(d.get("direct_count", 0)),
        "broadcast_count": int(d.get("broadcast_count", 0)),
    }
    for u, v, d in G.edges(data=True)
    if u in top_node_set and v in top_node_set
]

top_nodes_data = [
    {
        "id": n,
        "sent": d.get("sent", 0),
        "received": d.get("received", 0),
        "unique_contacts": d.get("unique_contacts", 0),
        "pagerank": round(d.get("pagerank", 0), 6),
        "betweenness": round(d.get("betweenness", 0), 6),
        "community": d.get("community", -1),
        "role": d.get("inferred_role", "Unknown"),
    }
    for n, d in sorted(
        G.nodes(data=True), key=lambda x: x[1].get("pagerank", 0), reverse=True
    )[:30]
]

comm_members = defaultdict(list)
for n in top_node_ids:
    comm_members[G.nodes[n].get("community", -1)].append(n)
community_summary = [
    {
        "community_id": cid,
        "size": len(ms),
        "top_members": sorted(
            ms, key=lambda n: G.nodes[n].get("pagerank", 0), reverse=True
        )[:5],
    }
    for cid, ms in sorted(comm_members.items(), key=lambda x: -len(x[1]))
]

stats = {
    "total_emails": total_ok,
    "total_nodes": G.number_of_nodes(),
    "total_edges": G.number_of_edges(),
    "n_communities": n_communities,
    "density": round(nx.density(G), 6),
    "avg_clustering": round(nx.average_clustering(Gcc), 4),
    "lcc_nodes": Gcc.number_of_nodes(),
    "lcc_edges": Gcc.number_of_edges(),
    "date_range_start": min((d for d in dates_list if d), default=""),
    "date_range_end": max((d for d in dates_list if d), default=""),
}

SEP = (",", ":")
with open(f"{OUTPUT_DIR}/graph_data.json", "w") as f:
    json.dump({"nodes": nodes_out, "edges": edges_out}, f, separators=SEP)
with open(f"{OUTPUT_DIR}/timeline.json", "w") as f:
    json.dump(timeline_data, f, separators=SEP)
with open(f"{OUTPUT_DIR}/keywords.json", "w") as f:
    json.dump(top_keywords, f, separators=SEP)
with open(f"{OUTPUT_DIR}/degree_dist.json", "w") as f:
    json.dump(degree_dist_data, f, separators=SEP)
with open(f"{OUTPUT_DIR}/top_nodes.json", "w") as f:
    json.dump(top_nodes_data, f, separators=SEP)
with open(f"{OUTPUT_DIR}/communities.json", "w") as f:
    json.dump(community_summary, f, separators=SEP)
with open(f"{OUTPUT_DIR}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n✅ Done! Output written to:", OUTPUT_DIR)
for k, v in stats.items():
    print(f"   {k}: {v}")
