#!/usr/bin/env python
"""
export_pharmgkb_layer.py
------------------------
Takes in pharmgkb_wide_enriched.tsv and pharmgkb_long_enriched.tsv

Output: 
  pharmgkb_graph_chunks/
    nodes_variant_001.json        -- variant nodes (rsid-keyed)
    nodes_gene_001.json           -- gene nodes enriched with GWAS gene-level stats
    nodes_drug_001.json           -- drug nodes from PharmGKB
    nodes_population_001.json     -- population nodes (one per ancestry group)
    edges_variant_gene_001.json   -- variant → gene edges with evidence + GWAS stats
    edges_variant_drug_001.json   -- variant → drug edges (one per exploded drug)
    edges_variant_pop_001.json    -- variant → population edges with allele frequency
  pharmgkb_layer.json             -- manifest consumed by index.html


Usage (run in the directory with pharmgkb_wide_enriched.tsv and pharmgkb_long_enriched.tsv):
    python export_pharmgkb_layer.py

"""

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_WIDE = BASE_DIR / "pharmgkb_wide_enriched.tsv"
INPUT_LONG = BASE_DIR / "pharmgkb_long_enriched.tsv"
OUTPUT_DIR = BASE_DIR / "pharmgkb_graph_chunks"
MANIFEST_JSON = BASE_DIR / "pharmgkb_layer.json"

MAX_NODES_PER_FILE = 3000
MAX_EDGES_PER_FILE = 5000

# ---------------------------------------------------------------------------
# Colour palette 
# ---------------------------------------------------------------------------
NODE_COLORS = {
    "variant":    "#a855f7",   # purple  
    "gene":       "#f59e0b",   # amber   
    "drug":       "#60a5fa",   # blue    
    "population": "#10b981",   # emerald 
}

# Population code → readable label
POP_LABELS = {
    "afr": "African/African American",
    "amr": "Admixed American",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "fin": "Finnish",
    "mid": "Middle Eastern",
    "nfe": "Non-Finnish European",
    "oth": "Other",
    "sas": "South Asian",
}

# Evidence level ordering (1A is highest)
EVIDENCE_ORDER = {"1A": 0, "1B": 1, "2A": 2, "2B": 3, "3": 4, "4": 5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    return s if s and s.lower() not in {"nan", "none", ""} else None


def safe_float(value):
    v = clean(value)
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def safe_bool(value):
    v = clean(value)
    if v is None:
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "t"}


def split_drugs(value):
    v = clean(value)
    if v is None:
        return []
    return [d.strip() for d in v.split(";") if d.strip()]


def node_id_variant(rsid):
    return f"variant:{rsid}"


def node_id_gene(symbol):
    return f"gene:{symbol.upper()}"


def node_id_drug(name):
    return f"drug:{name.lower().replace(' ', '_')}"


def node_id_pop(code):
    return f"population:{code}"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# Build node + edge collections
# ---------------------------------------------------------------------------
def build_graph(wide: pd.DataFrame, long_df: pd.DataFrame):
    """
    Returns:
        nodes   dict[node_id -> node_record]
        edges   list[edge_record]
    """
    nodes: dict = {}
    edges: list = []

    # ------------------------------------------------------------------
    # 1. Population nodes  (fixed set from POP_LABELS)
    # ------------------------------------------------------------------
    freq_cols_wide = [c for c in wide.columns if c.startswith("freq_")]
    pop_codes_in_data = sorted({c[5:] for c in freq_cols_wide})  # strip "freq_"

    for code in pop_codes_in_data:
        nid = node_id_pop(code)
        nodes[nid] = {
            "id": nid,
            "label": POP_LABELS.get(code, code.upper()),
            "type": "population",
            "group": "population",
            "color": NODE_COLORS["population"],
            "population_code": code,
            "layer": "pharmgkb",
        }

    # ------------------------------------------------------------------
    # 2. Per-row processing of the WIDE table
    # ------------------------------------------------------------------
    # Aggregate per-variant stats (GWAS + evidence) for variant nodes
    variant_meta: dict = defaultdict(lambda: {
        "genes": set(),
        "drugs": set(),
        "phenotypes": set(),
        "evidence_levels": set(),
        "gwas_p_values": [],
        "gwas_or": [],
        "gwas_beta": [],
        "gwas_traits": set(),
        "gwas_pubmeds": set(),
        "gwas_significant": False,
        "n_gwas_studies": 0,
        "n_gwas_significant": 0,
        "pharmgkb_p_values": [],
        "pharmgkb_sig_studies": 0,
    })

    gene_meta: dict = defaultdict(lambda: {
        "variants": set(),
        "drugs": set(),
        "phenotypes": set(),
        "evidence_levels": set(),
    })

    drug_meta: dict = defaultdict(lambda: {
        "variants": set(),
        "genes": set(),
        "phenotypes": set(),
        "evidence_levels": set(),
    })

    for _, row in wide.iterrows():
        rsid  = clean(row["rsid"])
        gene  = clean(row["gene"])
        drug_raw = clean(row["drug"])
        evidence = clean(row.get("evidence"))
        phenotype = clean(row.get("phenotype"))

        if not rsid or not gene:
            continue

        drugs = split_drugs(drug_raw)
        vid = node_id_variant(rsid)
        gid = node_id_gene(gene)

        # ---- variant meta ----
        vm = variant_meta[rsid]
        vm["genes"].add(gene)
        vm["drugs"].update(drugs)
        if phenotype:
            vm["phenotypes"].add(phenotype)
        if evidence:
            vm["evidence_levels"].add(str(evidence))
        gp = safe_float(row.get("best_gwas_p_value"))
        if gp is not None:
            vm["gwas_p_values"].append(gp)
        go = safe_float(row.get("best_gwas_odds_ratio"))
        if go is not None:
            vm["gwas_or"].append(go)
        gb = safe_float(row.get("best_gwas_beta"))
        if gb is not None:
            vm["gwas_beta"].append(gb)
        gt = clean(row.get("best_gwas_trait"))
        if gt:
            vm["gwas_traits"].add(gt)
        gpm = safe_float(row.get("best_gwas_pubmed_id"))
        if gpm is not None:
            vm["gwas_pubmeds"].add(str(int(gpm)))
        if safe_bool(row.get("gwas_significant")):
            vm["gwas_significant"] = True
        vm["n_gwas_studies"] = max(vm["n_gwas_studies"], int(row.get("n_gwas_studies") or 0))
        vm["n_gwas_significant"] = max(vm["n_gwas_significant"], int(row.get("n_gwas_significant") or 0))
        pp = safe_float(row.get("pharmgkb_best_p_value"))
        if pp is not None:
            vm["pharmgkb_p_values"].append(pp)
        vm["pharmgkb_sig_studies"] = max(vm["pharmgkb_sig_studies"], int(row.get("pharmgkb_sig_studies") or 0))

        # ---- gene meta ----
        gm = gene_meta[gene]
        gm["variants"].add(rsid)
        gm["drugs"].update(drugs)
        if phenotype:
            gm["phenotypes"].add(phenotype)
        if evidence:
            gm["evidence_levels"].add(str(evidence))

        # ---- drug meta ----
        for drug in drugs:
            dm = drug_meta[drug]
            dm["variants"].add(rsid)
            dm["genes"].add(gene)
            if phenotype:
                dm["phenotypes"].add(phenotype)
            if evidence:
                dm["evidence_levels"].add(str(evidence))

        # ---- variant → gene edge ----
        vg_edge_id = f"pharmgkb|variant_gene|{rsid}|{gene}"
        vg_edge = {
            "id": vg_edge_id,
            "source": vid,
            "target": gid,
            "source_type": "variant",
            "target_type": "gene",
            "edge_type": "variant_gene",
            "relation": "variant_gene",
            "source_label": rsid,
            "target_label": gene,
            "gene_symbol": [gene],
            "variant_name": [rsid],
            "evidence_level": [str(evidence)] if evidence else [],
            "phenotype_names": [phenotype] if phenotype else [],
            "gwas_significant": safe_bool(row.get("gwas_significant")),
            "best_gwas_p_value": gp,
            "best_gwas_odds_ratio": go,
            "best_gwas_beta": gb,
            "best_gwas_trait": gt,
            "n_gwas_studies": int(row.get("n_gwas_studies") or 0),
            "n_gwas_significant": int(row.get("n_gwas_significant") or 0),
            "pharmgkb_best_p_value": pp,
            "pharmgkb_sig_studies": int(row.get("pharmgkb_sig_studies") or 0),
            "source_database": ["pharmgkb"],
            "layer": "pharmgkb",
        }
        # Attach per-ancestry frequencies directly on the edge
        freq_dict = {}
        for fc in freq_cols_wide:
            code = fc[5:]
            fv = safe_float(row.get(fc))
            if fv is not None:
                freq_dict[code] = fv
        if freq_dict:
            vg_edge["allele_frequencies"] = freq_dict

        edges.append(vg_edge)

        # ---- variant → drug edges ----
        for drug in drugs:
            did = node_id_drug(drug)
            vd_edge = {
                "id": f"pharmgkb|variant_drug|{rsid}|{drug}",
                "source": vid,
                "target": did,
                "source_type": "variant",
                "target_type": "drug",
                "edge_type": "variant_drug",
                "relation": "variant_drug",
                "source_label": rsid,
                "target_label": drug,
                "gene_symbol": [gene],
                "variant_name": [rsid],
                "drug_name": [drug],
                "evidence_level": [str(evidence)] if evidence else [],
                "phenotype_names": [phenotype] if phenotype else [],
                "gwas_significant": safe_bool(row.get("gwas_significant")),
                "best_gwas_p_value": gp,
                "source_database": ["pharmgkb"],
                "layer": "pharmgkb",
            }
            edges.append(vd_edge)

    # ------------------------------------------------------------------
    # 3. Population-frequency edges from the long table
    # ------------------------------------------------------------------
    # One edge per (rsid, population_code) – aggregate across drugs/phenotypes
    pop_edge_accum: dict = defaultdict(lambda: {
        "genes": set(), "drugs": set(), "phenotypes": set(),
        "evidence_levels": set(), "frequencies": [],
    })

    for _, row in long_df.iterrows():
        rsid  = clean(row["rsid"])
        gene  = clean(row["gene"])
        drug_raw = clean(row["drug"])
        pop   = clean(row.get("population_code"))
        freq  = safe_float(row.get("frequency"))
        evidence = clean(row.get("evidence"))
        phenotype = clean(row.get("phenotype"))

        if not rsid or not pop:
            continue

        key = (rsid, pop)
        rec = pop_edge_accum[key]
        if gene:
            rec["genes"].add(gene)
        rec["drugs"].update(split_drugs(drug_raw) if drug_raw else [])
        if phenotype:
            rec["phenotypes"].add(phenotype)
        if evidence:
            rec["evidence_levels"].add(str(evidence))
        if freq is not None:
            rec["frequencies"].append(freq)

    for (rsid, pop), rec in pop_edge_accum.items():
        vid = node_id_variant(rsid)
        pid = node_id_pop(pop)
        freqs = rec["frequencies"]
        mean_freq = sum(freqs) / len(freqs) if freqs else None
        edges.append({
            "id": f"pharmgkb|variant_pop|{rsid}|{pop}",
            "source": vid,
            "target": pid,
            "source_type": "variant",
            "target_type": "population",
            "edge_type": "variant_population",
            "relation": "variant_population",
            "source_label": rsid,
            "target_label": POP_LABELS.get(pop, pop),
            "variant_name": [rsid],
            "gene_symbol": sorted(rec["genes"]),
            "drug_name": sorted(rec["drugs"]),
            "phenotype_names": sorted(rec["phenotypes"]),
            "evidence_level": sorted(rec["evidence_levels"]),
            "allele_frequency": mean_freq,
            "n_frequency_records": len(freqs),
            "population_code": pop,
            "population_name": POP_LABELS.get(pop, pop),
            "source_database": ["pharmgkb"],
            "layer": "pharmgkb",
        })

    # ------------------------------------------------------------------
    # 4. Build variant / gene / drug nodes
    # ------------------------------------------------------------------
    # Degree counts from edges
    degree_map: dict = defaultdict(int)
    for e in edges:
        degree_map[e["source"]] += 1
        degree_map[e["target"]] += 1

    # Variant nodes
    for rsid, vm in variant_meta.items():
        nid = node_id_variant(rsid)
        deg = degree_map[nid]
        gp_vals = vm["gwas_p_values"]
        best_p = min(gp_vals) if gp_vals else None
        pp_vals = vm["pharmgkb_p_values"]
        best_pp = min(pp_vals) if pp_vals else None
        ev_sorted = sorted(vm["evidence_levels"], key=lambda x: EVIDENCE_ORDER.get(x, 99))
        nodes[nid] = {
            "id": nid,
            "label": rsid,
            "type": "variant",
            "group": "variant",
            "color": NODE_COLORS["variant"],
            "degree": deg,
            "size": 2 + min(deg * 0.15, 20),
            "variant_name": rsid,
            "gene_symbol": sorted(vm["genes"]),
            "drug_name": sorted(vm["drugs"]),
            "phenotypes": sorted(vm["phenotypes"]),
            "evidence_levels": ev_sorted,
            "best_evidence": ev_sorted[0] if ev_sorted else None,
            "gwas_significant": vm["gwas_significant"],
            "best_gwas_p_value": best_p,
            "n_gwas_studies": vm["n_gwas_studies"],
            "n_gwas_significant": vm["n_gwas_significant"],
            "gwas_traits": sorted(vm["gwas_traits"]),
            "gwas_pubmeds": sorted(vm["gwas_pubmeds"]),
            "pharmgkb_best_p_value": best_pp,
            "pharmgkb_sig_studies": vm["pharmgkb_sig_studies"],
            "layer": "pharmgkb",
        }

    # Gene nodes
    for gene, gm in gene_meta.items():
        nid = node_id_gene(gene)
        deg = degree_map[nid]
        ev_sorted = sorted(gm["evidence_levels"], key=lambda x: EVIDENCE_ORDER.get(x, 99))
        nodes[nid] = {
            "id": nid,
            "label": gene,
            "type": "gene",
            "group": "gene",
            "color": NODE_COLORS["gene"],
            "degree": deg,
            "size": 2 + min(deg * 0.15, 20),
            "gene_symbol": gene,
            "variant_name": sorted(gm["variants"]),
            "drug_name": sorted(gm["drugs"]),
            "phenotypes": sorted(gm["phenotypes"]),
            "evidence_levels": ev_sorted,
            "best_evidence": ev_sorted[0] if ev_sorted else None,
            "layer": "pharmgkb",
        }

    # Drug nodes
    for drug, dm in drug_meta.items():
        nid = node_id_drug(drug)
        deg = degree_map[nid]
        ev_sorted = sorted(dm["evidence_levels"], key=lambda x: EVIDENCE_ORDER.get(x, 99))
        nodes[nid] = {
            "id": nid,
            "label": drug,
            "type": "drug",
            "group": "drug",
            "color": NODE_COLORS["drug"],
            "degree": deg,
            "size": 2 + min(deg * 0.15, 20),
            "drug_name": drug,
            "variant_name": sorted(dm["variants"]),
            "gene_symbol": sorted(dm["genes"]),
            "phenotypes": sorted(dm["phenotypes"]),
            "evidence_levels": ev_sorted,
            "best_evidence": ev_sorted[0] if ev_sorted else None,
            "layer": "pharmgkb",
        }

    # Update population node degrees
    for code in pop_codes_in_data:
        nid = node_id_pop(code)
        nodes[nid]["degree"] = degree_map.get(nid, 0)
        nodes[nid]["size"] = 2 + min(degree_map.get(nid, 0) * 0.15, 20)

    return nodes, edges


# ---------------------------------------------------------------------------
# Chunking + writing
# ---------------------------------------------------------------------------
def export_nodes(nodes: dict, output_dir: Path):
    by_type: dict = defaultdict(list)
    for node in nodes.values():
        by_type[node["type"]].append(node)

    manifest_entries = []
    for ntype in sorted(by_type):
        group = by_type[ntype]
        for part, chunk in enumerate(chunk_list(group, MAX_NODES_PER_FILE), start=1):
            fname = f"pharmgkb_nodes_{ntype}_{part:03d}.json"
            payload = {
                "chunk_type": "nodes",
                "node_type": ntype,
                "part": part,
                "count": len(chunk),
                "layer": "pharmgkb",
                "nodes": chunk,
            }
            write_json(output_dir / fname, payload)
            manifest_entries.append({
                "file": f"{output_dir.name}/{fname}",
                "node_type": ntype,
                "part": part,
                "count": len(chunk),
            })
    return manifest_entries


def export_edges(edges: list, output_dir: Path):
    by_pair: dict = defaultdict(list)
    for edge in edges:
        key = (edge.get("source_type", "unknown"), edge.get("target_type", "unknown"))
        by_pair[key].append(edge)

    manifest_entries = []
    for (src, tgt) in sorted(by_pair):
        group = by_pair[(src, tgt)]
        for part, chunk in enumerate(chunk_list(group, MAX_EDGES_PER_FILE), start=1):
            fname = f"pharmgkb_edges_{src}_to_{tgt}_{part:03d}.json"
            edge_types = sorted({e.get("edge_type", "unknown") for e in chunk})
            payload = {
                "chunk_type": "edges",
                "source_type": src,
                "target_type": tgt,
                "part": part,
                "count": len(chunk),
                "edge_types": edge_types,
                "layer": "pharmgkb",
                "links": chunk,
                "edges": chunk,
            }
            write_json(output_dir / fname, payload)
            manifest_entries.append({
                "file": f"{output_dir.name}/{fname}",
                "source_type": src,
                "target_type": tgt,
                "part": part,
                "count": len(chunk),
                "edge_types": edge_types,
            })
    return manifest_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not INPUT_WIDE.exists():
        raise FileNotFoundError(f"Missing: {INPUT_WIDE}")
    if not INPUT_LONG.exists():
        raise FileNotFoundError(f"Missing: {INPUT_LONG}")

    print("Reading input files...")
    wide = pd.read_csv(INPUT_WIDE, sep="\t", low_memory=False)
    long_df = pd.read_csv(INPUT_LONG, sep="\t", low_memory=False)

    print(f"  Wide table: {wide.shape[0]} rows × {wide.shape[1]} cols")
    print(f"  Long table: {long_df.shape[0]} rows × {long_df.shape[1]} cols")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building graph...")
    nodes, edges = build_graph(wide, long_df)

    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")

    print("Writing node chunks...")
    node_files = export_nodes(nodes, OUTPUT_DIR)

    print("Writing edge chunks...")
    edge_files = export_edges(edges, OUTPUT_DIR)

    # Collect summary stats
    edge_types = sorted({e.get("edge_type", "unknown") for e in edges})
    node_types = sorted({n["type"] for n in nodes.values()})

    manifest = {
        "metadata": {
            "graph_type": "pharmgkb enriched layer",
            "layer": "pharmgkb",
            "input_wide_file": INPUT_WIDE.name,
            "input_long_file": INPUT_LONG.name,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "edge_types": edge_types,
            "node_types": node_types,
            "chunk_output_dir": OUTPUT_DIR.name,
            "max_nodes_per_file": MAX_NODES_PER_FILE,
            "max_edges_per_file": MAX_EDGES_PER_FILE,
            "population_codes": list(POP_LABELS.keys()),
            "evidence_levels": ["1A", "1B", "2A", "2B", "3", "4"],
        },
        "node_files": node_files,
        "edge_files": edge_files,
    }

    write_json(MANIFEST_JSON, manifest)

    print(f"\nDone.")
    print(f"  Manifest  : {MANIFEST_JSON}")
    print(f"  Chunks dir: {OUTPUT_DIR}")
    print(f"  Node files: {len(node_files)}")
    print(f"  Edge files: {len(edge_files)}")
    print(f"  Edge types: {edge_types}")
    print(f"  Node types: {node_types}")


if __name__ == "__main__":
    main()
