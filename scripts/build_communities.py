#!/usr/bin/env python3
"""Build a unified knowledge graph over records, entities, and geo-correlations.

Runs Louvain community detection + centrality metrics and writes
ui/graph_communities.json consumed by ui/index.html for community-coloured
node rendering.

Inputs:
  ui/entities.json      — NER entities per record (Stream C)
  ui/citations.json     — resolved citation edges (Stream D)
  ui/correlations.json  — geo/temporal correlation clusters (Stream K)

Output:
  ui/graph_communities.json
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms import (
    degree_centrality,
    betweenness_centrality,
    eigenvector_centrality,
)

ROOT = Path(__file__).resolve().parent.parent
ENTITIES_PATH  = ROOT / "ui" / "entities.json"
CITATIONS_PATH = ROOT / "ui" / "citations.json"
CORR_PATH      = ROOT / "ui" / "correlations.json"
OUT_PATH       = ROOT / "ui" / "graph_communities.json"

MAX_CITATION_WEIGHT = 5.0


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> nx.Graph:
    G = nx.Graph()

    # ── 1. record–entity edges from entities.json ──────────────────────────
    entities_data = json.loads(ENTITIES_PATH.read_text())
    by_record = entities_data.get("by_record", {})

    for record_id, entity_list in by_record.items():
        G.add_node(record_id, node_type="record")
        for ent in entity_list:
            G.add_node(ent, node_type="entity")
            if G.has_edge(record_id, ent):
                G[record_id][ent]["weight"] += 1
            else:
                G.add_edge(record_id, ent, weight=1.0)

    # ── 2. record–record edges from citations.json ────────────────────────
    citations_data = json.loads(CITATIONS_PATH.read_text())
    for edge in citations_data.get("edges", []):
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        count = min(float(edge.get("count", 1) or 1), MAX_CITATION_WEIGHT)
        for nid in (src, dst):
            if nid not in G:
                G.add_node(nid, node_type="record")
        if G.has_edge(src, dst):
            G[src][dst]["weight"] = min(G[src][dst]["weight"] + count, MAX_CITATION_WEIGHT)
        else:
            G.add_edge(src, dst, weight=count)

    # ── 3. record–record / record–geocluster from correlations.json ───────
    corr_data = json.loads(CORR_PATH.read_text())
    for official_id, cluster in corr_data.items():
        official_node = cluster.get("official", {}).get("id") or official_id
        # Normalise: strip "wargov:" prefix if present so IDs match records
        if ":" in official_node:
            official_node = official_node.split(":", 1)[1]

        # Build a geocluster node for this correlation group
        geo_node = f"geo:{official_id}"
        G.add_node(geo_node, node_type="geocluster")

        match_count = cluster.get("match_count", 0) or 0
        base_weight = max(0.1, min(1.0, match_count / 5.0))

        # Link each matched civilian record to the geo-cluster
        for match in cluster.get("matches", []):
            match_id = match.get("id")
            if not match_id:
                continue
            dist_km = match.get("distance_km") or 150.0
            delta_days = match.get("delta_days") or 30
            # Score: closer and more recent = higher weight
            dist_score = max(0.0, 1.0 - dist_km / 150.0)
            time_score = max(0.0, 1.0 - delta_days / 30.0)
            w = max(0.05, (dist_score + time_score) / 2.0)
            if match_id not in G:
                G.add_node(match_id, node_type="record")
            if G.has_edge(geo_node, match_id):
                G[geo_node][match_id]["weight"] = max(G[geo_node][match_id]["weight"], w)
            else:
                G.add_edge(geo_node, match_id, weight=w)

        # Also link official record → geo-cluster
        if official_node not in G:
            G.add_node(official_node, node_type="record")
        if G.has_edge(official_node, geo_node):
            G[official_node][geo_node]["weight"] = max(G[official_node][geo_node]["weight"], base_weight)
        else:
            G.add_edge(official_node, geo_node, weight=base_weight)

    return G


# ─── Community label ──────────────────────────────────────────────────────────

def community_label(community_nodes: set, G: nx.Graph) -> str:
    """Pick a human-readable label from the top-3 highest-degree entity nodes."""
    entity_nodes = [n for n in community_nodes if G.nodes[n].get("node_type") == "entity"]
    if not entity_nodes:
        # Fall back to top-degree record nodes
        entity_nodes = list(community_nodes)
    entity_nodes.sort(key=lambda n: G.degree(n), reverse=True)
    top3 = entity_nodes[:3]
    # Clean up the entity surface — strip prefix like "org:", "person:", etc.
    def clean(s):
        if ":" in s:
            s = s.split(":", 1)[1]
        return s.replace("_", " ").strip()
    return " / ".join(clean(n) for n in top3)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("  building unified knowledge graph…")
    G = build_graph()
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Louvain community detection ────────────────────────────────────────
    print("  running Louvain community detection (resolution=1.0)…")
    # louvain_communities returns a list of frozensets
    communities_raw = louvain_communities(G, weight="weight", resolution=1.0, seed=42)
    communities_raw = sorted(communities_raw, key=len, reverse=True)  # largest first

    # Build by_node map and community records
    by_node: dict[str, int] = {}
    communities_out = []
    for cid, members in enumerate(communities_raw):
        for node in members:
            by_node[node] = cid
        label = community_label(set(members), G)
        sample = sorted(
            [n for n in members if G.nodes[n].get("node_type") == "record"],
            key=lambda n: G.degree(n), reverse=True
        )[:5]
        communities_out.append({
            "id": cid,
            "label": label,
            "size": len(members),
            "members_sample": sample,
        })

    # ── Modularity ────────────────────────────────────────────────────────
    from networkx.algorithms.community.quality import modularity
    mod = modularity(G, communities_raw, weight="weight")
    print(f"  modularity: {mod:.4f}  communities: {len(communities_raw)}")

    # ── Centrality metrics ────────────────────────────────────────────────
    print("  computing degree centrality…")
    deg_cent = degree_centrality(G)

    # Betweenness: only top-100 nodes by degree (full is too expensive)
    top100_nodes = sorted(deg_cent, key=deg_cent.get, reverse=True)[:100]
    subG = G.subgraph(top100_nodes)
    print("  computing betweenness centrality on top-100 nodes…")
    btw_cent_sub = betweenness_centrality(subG, weight="weight", normalized=True)
    # Fill zeros for the rest
    btw_cent: dict[str, float] = {n: 0.0 for n in G.nodes()}
    btw_cent.update(btw_cent_sub)

    print("  computing eigenvector centrality…")
    try:
        eig_cent = eigenvector_centrality(G, max_iter=500, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # Fallback: degree-normalised proxy
        max_deg = max(deg_cent.values()) or 1
        eig_cent = {n: v / max_deg for n, v in deg_cent.items()}

    # Combine
    centrality: dict[str, dict] = {}
    for n in G.nodes():
        centrality[n] = {
            "degree":      round(deg_cent.get(n, 0.0), 6),
            "betweenness": round(btw_cent.get(n, 0.0), 6),
            "eigenvector": round(eig_cent.get(n, 0.0), 6),
        }

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = {
        "total_nodes":  G.number_of_nodes(),
        "total_edges":  G.number_of_edges(),
        "modularity":   round(mod, 6),
        "n_communities": len(communities_raw),
    }

    # ── Write output ──────────────────────────────────────────────────────
    payload = {
        "communities": communities_out,
        "by_node":     by_node,
        "centrality":  centrality,
        "stats":       stats,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"  → {OUT_PATH}")

    # ── Smoke-test report ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  modularity:       {mod:.4f}")
    print(f"  communities:      {len(communities_raw)}")
    print()
    print("  Top-5 communities by size:")
    for c in communities_out[:5]:
        print(f"    [{c['id']:3d}] size={c['size']:5d}  {c['label']}")
    print()
    top5_eig = sorted(eig_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top-5 nodes by eigenvector centrality:")
    for nid, val in top5_eig:
        nt = G.nodes[nid].get("node_type", "?")
        print(f"    {val:.4f}  [{nt}]  {nid}")
    print("=" * 60)


if __name__ == "__main__":
    main()
