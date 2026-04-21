import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_EDGES = BASE_DIR / "full_interaction_edges.csv"
INPUT_NODES = BASE_DIR / "full_interaction_nodes.csv"

OUTPUT_DIR = BASE_DIR / "full_graph_chunks"
MANIFEST_JSON = BASE_DIR / "full_interaction_graph.json"

MAX_NODES_PER_FILE = 3000
MAX_EDGES_PER_FILE = 5000

NODE_COLORS = {
    "gene": "#ff7f0e",
    "drug": "#4da6ff",
    "variant": "#2ca02c",
    "disease": "#d62728",
}


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def parse_bool(value):
    value = clean_value(value)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def parse_number(value):
    value = clean_value(value)
    if value is None:
        return None
    try:
        num = float(value)
        if num.is_integer():
            return int(num)
        return num
    except Exception:
        return value


def parse_multi_value(value):
    value = clean_value(value)
    if value is None:
        return []

    text = str(value)
    text = text.replace(";", "|")
    parts = [item.strip() for item in text.split("|") if item.strip()]

    seen = set()
    out = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            out.append(part)
    return out


def add_if_not_none(d, key, value):
    if value is not None:
        d[key] = value


def normalize_text(value, fallback="unknown"):
    value = clean_value(value)
    if value is None:
        return fallback
    return str(value).strip()


def slugify(value):
    text = normalize_text(value, "unknown").lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def chunk_list(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def build_node_record(row, degree_map):
    node_id = clean_value(row.get("node_id"))
    node_name = clean_value(row.get("node_name"))
    node_type = clean_value(row.get("node_type"))

    degree = degree_map.get(node_id, 0)
    size = 2 + min(degree * 0.15, 20)

    node = {
        "id": node_id,
        "label": node_name if node_name is not None else node_id,
        "type": node_type,
        "group": node_type,
        "degree": degree,
        "size": size,
        "color": NODE_COLORS.get(str(node_type).lower() if node_type else "", "#999999"),
    }

    add_if_not_none(node, "gene_symbol", clean_value(row.get("gene_symbol")))
    add_if_not_none(node, "gene_name", clean_value(row.get("gene_name")))
    add_if_not_none(node, "gene_family", parse_multi_value(row.get("gene_family")))
    add_if_not_none(node, "primary_gene_family", parse_multi_value(row.get("primary_gene_family")))
    add_if_not_none(node, "gene_family_id", parse_multi_value(row.get("gene_family_id")))
    add_if_not_none(node, "parent_gene_family_ids", parse_multi_value(row.get("parent_gene_family_ids")))
    add_if_not_none(node, "parent_gene_families", parse_multi_value(row.get("parent_gene_families")))
    add_if_not_none(node, "variant_name", clean_value(row.get("variant_name")))
    add_if_not_none(node, "drug_name", clean_value(row.get("drug_name")))
    add_if_not_none(node, "disease_name", clean_value(row.get("disease_name")))
    add_if_not_none(node, "connected_edge_types", parse_multi_value(row.get("connected_edge_types")))

    return node


def build_link_record(row, idx):
    source = clean_value(row.get("source_node_id"))
    target = clean_value(row.get("target_node_id"))
    source_type = clean_value(row.get("source_type"))
    target_type = clean_value(row.get("target_type"))
    edge_type = clean_value(row.get("edge_type"))

    link = {
        "id": clean_value(row.get("edge_key")) or f"link_{idx}",
        "source": source,
        "target": target,
        "relation": edge_type,
        "source_type": source_type,
        "target_type": target_type,
        "edge_type": edge_type,
        "source_label": clean_value(row.get("source")),
        "target_label": clean_value(row.get("target")),
    }

    add_if_not_none(link, "source_raw_ids", parse_multi_value(row.get("source_raw_ids")))
    add_if_not_none(link, "target_raw_ids", parse_multi_value(row.get("target_raw_ids")))
    add_if_not_none(link, "gene_symbol", parse_multi_value(row.get("gene_symbol")))
    add_if_not_none(link, "gene_name", parse_multi_value(row.get("gene_name")))
    add_if_not_none(link, "gene_family", parse_multi_value(row.get("gene_family")))
    add_if_not_none(link, "primary_gene_family", parse_multi_value(row.get("primary_gene_family")))
    add_if_not_none(link, "gene_family_id", parse_multi_value(row.get("gene_family_id")))
    add_if_not_none(link, "parent_gene_family_ids", parse_multi_value(row.get("parent_gene_family_ids")))
    add_if_not_none(link, "parent_gene_families", parse_multi_value(row.get("parent_gene_families")))
    add_if_not_none(link, "gene_family_match_source", parse_multi_value(row.get("gene_family_match_source")))
    add_if_not_none(link, "drug_name", parse_multi_value(row.get("drug_name")))
    add_if_not_none(link, "variant_name", parse_multi_value(row.get("variant_name")))
    add_if_not_none(link, "disease_name", parse_multi_value(row.get("disease_name")))
    add_if_not_none(link, "phenotype_names", parse_multi_value(row.get("phenotype_names")))
    add_if_not_none(link, "evidence_level", parse_multi_value(row.get("evidence_level")))
    add_if_not_none(link, "evidence_type", parse_multi_value(row.get("evidence_type")))
    add_if_not_none(link, "association", parse_multi_value(row.get("association")))
    add_if_not_none(link, "clinical_annotation_type", parse_multi_value(row.get("clinical_annotation_type")))
    add_if_not_none(link, "interaction_type", parse_multi_value(row.get("interaction_type")))
    add_if_not_none(link, "source_database", parse_multi_value(row.get("source_database")))
    add_if_not_none(link, "source_version", parse_multi_value(row.get("source_version")))
    add_if_not_none(link, "pmids", parse_multi_value(row.get("pmids")))

    add_if_not_none(link, "interaction_score_max", parse_number(row.get("interaction_score_max")))
    add_if_not_none(link, "interaction_score_mean", parse_number(row.get("interaction_score_mean")))
    add_if_not_none(link, "pmid_count", parse_number(row.get("pmid_count")))
    add_if_not_none(link, "phenotype_count", parse_number(row.get("phenotype_count")))
    add_if_not_none(link, "drug_count_context", parse_number(row.get("drug_count_context")))
    add_if_not_none(link, "row_count", parse_number(row.get("row_count")))

    add_if_not_none(link, "pk_related", parse_bool(row.get("pk_related")))
    add_if_not_none(link, "pd_related", parse_bool(row.get("pd_related")))
    add_if_not_none(link, "approved", parse_bool(row.get("approved")))
    add_if_not_none(link, "immunotherapy", parse_bool(row.get("immunotherapy")))
    add_if_not_none(link, "anti_neoplastic", parse_bool(row.get("anti_neoplastic")))
    add_if_not_none(link, "has_gene_family", parse_bool(row.get("has_gene_family")))
    add_if_not_none(link, "has_phenotype", parse_bool(row.get("has_phenotype")))

    return link


def export_node_chunks(nodes):
    grouped = defaultdict(list)
    for node in nodes:
        node_type = normalize_text(node.get("type"))
        grouped[node_type].append(node)

    manifest_entries = []

    for node_type in sorted(grouped.keys()):
        group_nodes = grouped[node_type]

        for part_idx, chunk in enumerate(chunk_list(group_nodes, MAX_NODES_PER_FILE), start=1):
            file_name = f"nodes_{slugify(node_type)}_{part_idx:03d}.json"
            rel_path = f"{OUTPUT_DIR.name}/{file_name}"

            payload = {
                "chunk_type": "nodes",
                "node_type": node_type,
                "part": part_idx,
                "count": len(chunk),
                "nodes": chunk,
            }
            write_json(OUTPUT_DIR / file_name, payload)

            manifest_entries.append({
                "file": rel_path,
                "node_type": node_type,
                "part": part_idx,
                "count": len(chunk),
            })

    return manifest_entries


def export_edge_chunks(links):
    grouped = defaultdict(list)

    for link in links:
        source_type = normalize_text(link.get("source_type"))
        target_type = normalize_text(link.get("target_type"))
        grouped[(source_type, target_type)].append(link)

    manifest_entries = []

    for (source_type, target_type) in sorted(grouped.keys()):
        group_links = grouped[(source_type, target_type)]

        for part_idx, chunk in enumerate(chunk_list(group_links, MAX_EDGES_PER_FILE), start=1):
            file_name = (
                f"edges_{slugify(source_type)}_to_{slugify(target_type)}_{part_idx:03d}.json"
            )
            rel_path = f"{OUTPUT_DIR.name}/{file_name}"

            payload = {
                "chunk_type": "edges",
                "source_type": source_type,
                "target_type": target_type,
                "part": part_idx,
                "count": len(chunk),
                "edge_types": sorted(
                    {normalize_text(x.get('edge_type')) for x in chunk if clean_value(x.get("edge_type")) is not None}
                ),
                "links": chunk,
                "edges": chunk,
            }
            write_json(OUTPUT_DIR / file_name, payload)

            manifest_entries.append({
                "file": rel_path,
                "source_type": source_type,
                "target_type": target_type,
                "part": part_idx,
                "count": len(chunk),
                "edge_types": payload["edge_types"],
            })

    return manifest_entries


def main():
    if not INPUT_EDGES.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_EDGES}")
    if not INPUT_NODES.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_NODES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    edges_df = pd.read_csv(INPUT_EDGES, low_memory=False)
    nodes_df = pd.read_csv(INPUT_NODES, low_memory=False)

    degree_map = {}
    for _, row in edges_df.iterrows():
        source = clean_value(row.get("source_node_id"))
        target = clean_value(row.get("target_node_id"))
        if source:
            degree_map[source] = degree_map.get(source, 0) + 1
        if target:
            degree_map[target] = degree_map.get(target, 0) + 1

    nodes = []
    seen_nodes = set()
    for _, row in nodes_df.iterrows():
        node = build_node_record(row, degree_map)
        node_id = node["id"]
        if node_id and node_id not in seen_nodes:
            seen_nodes.add(node_id)
            nodes.append(node)

    links = []
    for idx, row in edges_df.iterrows():
        source = clean_value(row.get("source_node_id"))
        target = clean_value(row.get("target_node_id"))
        if not source or not target:
            continue
        links.append(build_link_record(row, idx))

    metadata = {
        "graph_type": "full interaction knowledge graph",
        "input_edges_file": INPUT_EDGES.name,
        "input_nodes_file": INPUT_NODES.name,
        "node_count": len(nodes),
        "link_count": len(links),
        "edge_types": sorted(
            [x for x in edges_df["edge_type"].dropna().astype(str).unique().tolist()]
        ),
        "node_types": sorted(
            [x for x in nodes_df["node_type"].dropna().astype(str).unique().tolist()]
        ),
        "chunk_output_dir": OUTPUT_DIR.name,
        "max_nodes_per_file": MAX_NODES_PER_FILE,
        "max_edges_per_file": MAX_EDGES_PER_FILE,
    }

    node_files = export_node_chunks(nodes)
    edge_files = export_edge_chunks(links)

    manifest = {
        "metadata": metadata,
        "node_files": node_files,
        "edge_files": edge_files,
    }

    write_json(MANIFEST_JSON, manifest)

    print(f"Saved manifest to: {MANIFEST_JSON}")
    print(f"Saved chunk files to: {OUTPUT_DIR}")
    print(f"Nodes: {len(nodes)}")
    print(f"Links: {len(links)}")
    print(f"Node chunk files: {len(node_files)}")
    print(f"Edge chunk files: {len(edge_files)}")


if __name__ == "__main__":
    main()