import os
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd

BASE_DIR = "."

INPUT_FILES = {
    "nodes": "unified_graph_nodes.csv",
    "edges": "unified_graph_edges.csv",
    "clinical": "clinicalVariants.tsv",
    "hgnc": "hgnc_complete_set.tsv",
    "hierarchy": "hierarchy.csv",
    "interactions": "interactions.tsv",
    "relationships": "relationships.tsv",
    "summary": "summary_annotations.tsv",
}

OUTPUT_FILES = {
    "nodes": "full_interaction_nodes.csv",
    "edges": "full_interaction_edges.csv",
}


# -----------------------------
# Helpers
# -----------------------------
def norm(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value if value else None


def split_pipe(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if str(x).strip()]


def split_variant_tokens(value):
    if pd.isna(value):
        return []
    return [
        x.strip()
        for x in re.split(r"\s*,\s*|\s*;\s*", str(value))
        if str(x).strip()
    ]


def split_drug_tokens(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in re.split(r"\s*;\s*|\s*\|\s*", str(value)) if str(x).strip()]


def split_pmids(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in re.split(r"\s*;\s*|\s*,\s*", str(value)) if str(x).strip()]


def join_unique(values, sep="|"):
    cleaned = []
    seen = set()
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        value = str(value).strip()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            cleaned.append(value)
    return sep.join(cleaned) if cleaned else np.nan


def to_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in {"1", "true", "t", "yes", "y"}


def looks_like_identifier(text):
    if text is None or pd.isna(text):
        return False
    text = str(text).strip()
    lowered = text.lower()
    return (
        lowered.startswith(("chembl:", "ncit:", "drugbank:", "cid:", "pubchem:", "inchikey:"))
        or re.fullmatch(r"pa\d+", lowered) is not None
    )


def choose_preferred_label(*candidates):
    valid = []
    for value in candidates:
        if pd.isna(value):
            continue
        value = str(value).strip()
        if value:
            valid.append(value)
    if not valid:
        return np.nan
    for value in valid:
        if not looks_like_identifier(value):
            return value
    return valid[0]


def safe_int_str(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def make_support_record():
    return {
        "source_raw_ids": set(),
        "target_raw_ids": set(),
        "phenotype_names": set(),
        "evidence_level": set(),
        "evidence_type": set(),
        "association": set(),
        "clinical_annotation_type": set(),
        "interaction_type": set(),
        "source_database": set(),
        "source_version": set(),
        "pmids": set(),
        "pmid_count_numeric": 0,
        "interaction_scores": [],
        "pk_related": False,
        "pd_related": False,
        "approved": False,
        "immunotherapy": False,
        "anti_neoplastic": False,
        "row_count": 0,
    }


def add_support(support_dict, pair_key, **kwargs):
    record = support_dict[pair_key]
    record["row_count"] += 1

    for field in [
        "source_raw_ids",
        "target_raw_ids",
        "phenotype_names",
        "evidence_level",
        "evidence_type",
        "association",
        "clinical_annotation_type",
        "interaction_type",
        "source_database",
        "source_version",
        "pmids",
    ]:
        values = kwargs.get(field, [])
        if values is None:
            continue
        if isinstance(values, (list, tuple, set)):
            for value in values:
                if value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip():
                    record[field].add(str(value).strip())
        else:
            if not pd.isna(values) and str(values).strip():
                record[field].add(str(values).strip())

    pmid_count = kwargs.get("pmid_count_numeric")
    if pmid_count is not None and not pd.isna(pmid_count):
        try:
            record["pmid_count_numeric"] += int(float(pmid_count))
        except Exception:
            pass

    interaction_score = kwargs.get("interaction_score")
    if interaction_score is not None and not pd.isna(interaction_score):
        try:
            record["interaction_scores"].append(float(interaction_score))
        except Exception:
            pass

    for bool_field in ["pk_related", "pd_related", "approved", "immunotherapy", "anti_neoplastic"]:
        record[bool_field] = record[bool_field] or to_bool(kwargs.get(bool_field))


# -----------------------------
# Read input
# -----------------------------
def read_inputs(base_dir):
    nodes = pd.read_csv(os.path.join(base_dir, INPUT_FILES["nodes"]), low_memory=False)
    edges = pd.read_csv(os.path.join(base_dir, INPUT_FILES["edges"]), low_memory=False)
    clinical = pd.read_csv(os.path.join(base_dir, INPUT_FILES["clinical"]), sep="\t", low_memory=False)
    hgnc = pd.read_csv(os.path.join(base_dir, INPUT_FILES["hgnc"]), sep="\t", low_memory=False)
    hierarchy = pd.read_csv(os.path.join(base_dir, INPUT_FILES["hierarchy"]), low_memory=False)
    interactions = pd.read_csv(os.path.join(base_dir, INPUT_FILES["interactions"]), sep="\t", low_memory=False)
    relationships = pd.read_csv(os.path.join(base_dir, INPUT_FILES["relationships"]), sep="\t", low_memory=False)
    summary = pd.read_csv(os.path.join(base_dir, INPUT_FILES["summary"]), sep="\t", low_memory=False)
    return nodes, edges, clinical, hgnc, hierarchy, interactions, relationships, summary


# -----------------------------
# Build lookup tables
# -----------------------------
def build_family_maps(hgnc, hierarchy):
    family_name_by_id = {}
    for _, row in hgnc.iterrows():
        family_ids = [safe_int_str(x) for x in split_pipe(row.get("gene_group_id"))]
        family_names = split_pipe(row.get("gene_group"))
        for gid, gname in zip(family_ids, family_names):
            if gid and gname and gid not in family_name_by_id:
                family_name_by_id[gid] = gname

    parent_map = defaultdict(set)
    for _, row in hierarchy.iterrows():
        child = safe_int_str(row.get("child_fam_id"))
        parent = safe_int_str(row.get("parent_fam_id"))
        if child and parent:
            parent_map[child].add(parent)

    ancestor_cache = {}

    def get_ancestors(family_ids):
        result = set()
        queue = deque([fid for fid in family_ids if fid])
        while queue:
            current = queue.popleft()
            if current in ancestor_cache:
                result.update(ancestor_cache[current])
                continue
            for parent in parent_map.get(current, set()):
                if parent not in result:
                    result.add(parent)
                    queue.append(parent)
        return sorted(result, key=lambda x: (len(x), x))

    return family_name_by_id, get_ancestors


def build_gene_lookup(hgnc, family_name_by_id, get_ancestors):
    priority = {"symbol": 0, "prev_symbol": 1, "alias_symbol": 2}
    gene_lookup = {}

    for _, row in hgnc.iterrows():
        symbol = row.get("symbol")
        canonical_key = norm(symbol)
        if not canonical_key:
            continue

        gene_name = row.get("name")
        family_ids = [safe_int_str(x) for x in split_pipe(row.get("gene_group_id")) if safe_int_str(x)]
        family_names = split_pipe(row.get("gene_group"))
        primary_family = family_names[0] if family_names else np.nan
        parent_ids = get_ancestors(family_ids)
        parent_names = [family_name_by_id.get(fid, fid) for fid in parent_ids]

        meta = {
            "node_name": symbol,
            "gene_symbol": symbol,
            "gene_name": gene_name,
            "gene_family": join_unique(family_names),
            "primary_gene_family": primary_family if primary_family else np.nan,
            "gene_family_id": join_unique(family_ids),
            "parent_gene_family_ids": join_unique(parent_ids),
            "parent_gene_families": join_unique(parent_names),
        }

        key_sources = {"symbol": [symbol]}
        key_sources["prev_symbol"] = split_pipe(row.get("prev_symbol"))
        key_sources["alias_symbol"] = split_pipe(row.get("alias_symbol"))

        for match_source, values in key_sources.items():
            for value in values:
                key = norm(value)
                if not key:
                    continue
                existing = gene_lookup.get(key)
                if existing is None or priority[match_source] < priority[existing["gene_family_match_source"]]:
                    gene_lookup[key] = {**meta, "gene_family_match_source": match_source}

    return gene_lookup


def build_drug_lookup(interactions, relationships, clinical, summary):
    accum = defaultdict(lambda: {"display_names": [], "raw_ids": []})

    def register(key, display_name=None, raw_id=None):
        key = norm(key)
        if not key:
            return
        if display_name is not None and not pd.isna(display_name) and str(display_name).strip():
            accum[key]["display_names"].append(str(display_name).strip())
        if raw_id is not None and not pd.isna(raw_id) and str(raw_id).strip():
            accum[key]["raw_ids"].append(str(raw_id).strip())

    for _, row in interactions.iterrows():
        label = choose_preferred_label(row.get("drug_claim_name"), row.get("drug_name"))
        register(row.get("drug_name"), label, row.get("drug_concept_id"))
        register(row.get("drug_claim_name"), label, row.get("drug_concept_id"))

    for _, row in relationships.iterrows():
        if row.get("Entity1_type") == "Chemical":
            register(row.get("Entity1_name"), row.get("Entity1_name"), row.get("Entity1_id"))
        if row.get("Entity2_type") == "Chemical":
            register(row.get("Entity2_name"), row.get("Entity2_name"), row.get("Entity2_id"))

    for _, row in clinical.iterrows():
        for drug in split_drug_tokens(row.get("chemicals")):
            register(drug, drug, None)

    for _, row in summary.iterrows():
        for drug in split_drug_tokens(row.get("Drug(s)")):
            register(drug, drug, None)

    drug_lookup = {}
    for key, value in accum.items():
        preferred = choose_preferred_label(*value["display_names"], key)
        drug_lookup[key] = {
            "node_name": preferred,
            "drug_name": preferred,
            "drug_raw_ids": join_unique(value["raw_ids"]),
        }
    return drug_lookup


def build_disease_lookup(relationships):
    accum = defaultdict(lambda: {"display_names": [], "raw_ids": []})

    def register(key, display_name=None, raw_id=None):
        key = norm(key)
        if not key:
            return
        if display_name is not None and not pd.isna(display_name) and str(display_name).strip():
            accum[key]["display_names"].append(str(display_name).strip())
        if raw_id is not None and not pd.isna(raw_id) and str(raw_id).strip():
            accum[key]["raw_ids"].append(str(raw_id).strip())

    for _, row in relationships.iterrows():
        if row.get("Entity1_type") == "Disease":
            register(row.get("Entity1_id"), row.get("Entity1_name"), row.get("Entity1_id"))
            register(row.get("Entity1_name"), row.get("Entity1_name"), row.get("Entity1_id"))
        if row.get("Entity2_type") == "Disease":
            register(row.get("Entity2_id"), row.get("Entity2_name"), row.get("Entity2_id"))
            register(row.get("Entity2_name"), row.get("Entity2_name"), row.get("Entity2_id"))

    disease_lookup = {}
    for key, value in accum.items():
        preferred = choose_preferred_label(*value["display_names"], key)
        disease_lookup[key] = {
            "node_name": preferred,
            "disease_name": preferred,
            "disease_raw_ids": join_unique(value["raw_ids"]),
        }
    return disease_lookup


def build_variant_lookup(clinical, summary, relationships):
    accum = defaultdict(lambda: {"display_names": [], "phenotypes": []})

    def register(key, display_name=None, phenotype=None):
        key = norm(key)
        if not key:
            return
        if display_name is not None and not pd.isna(display_name) and str(display_name).strip():
            accum[key]["display_names"].append(str(display_name).strip())
        if phenotype is not None and not pd.isna(phenotype) and str(phenotype).strip():
            accum[key]["phenotypes"].append(str(phenotype).strip())

    for _, row in clinical.iterrows():
        for variant in split_variant_tokens(row.get("variant")):
            register(variant, variant, row.get("phenotypes"))

    for _, row in summary.iterrows():
        for variant in split_variant_tokens(row.get("Variant/Haplotypes")):
            register(variant, variant, row.get("Phenotype(s)"))

    for _, row in relationships.iterrows():
        if row.get("Entity1_type") == "Variant":
            register(row.get("Entity1_name"), row.get("Entity1_name"), None)
        if row.get("Entity2_type") == "Variant":
            register(row.get("Entity2_name"), row.get("Entity2_name"), None)

    variant_lookup = {}
    for key, value in accum.items():
        preferred = choose_preferred_label(*value["display_names"], key)
        variant_lookup[key] = {
            "node_name": preferred,
            "variant_name": preferred,
            "variant_phenotypes": join_unique(value["phenotypes"]),
        }
    return variant_lookup


# -----------------------------
# Aggregate edge support from source tables
# -----------------------------
def build_edge_support(interactions, relationships, clinical, summary):
    gene_drug = defaultdict(make_support_record)
    gene_disease = defaultdict(make_support_record)
    variant_gene = defaultdict(make_support_record)
    variant_drug = defaultdict(make_support_record)

    # DGIdb-like gene-drug interactions
    for _, row in interactions.iterrows():
        gene_keys = {norm(row.get("gene_name")), norm(row.get("gene_claim_name"))}
        drug_keys = {norm(row.get("drug_name")), norm(row.get("drug_claim_name"))}
        gene_keys.discard(None)
        drug_keys.discard(None)
        for g in gene_keys:
            for d in drug_keys:
                pair_key = f"{g}|||{d}"
                add_support(
                    gene_drug,
                    pair_key,
                    source_raw_ids=[row.get("gene_concept_id")],
                    target_raw_ids=[row.get("drug_concept_id")],
                    interaction_type=[row.get("interaction_type")],
                    source_database=[row.get("interaction_source_db_name")],
                    source_version=[row.get("interaction_source_db_version")],
                    interaction_score=row.get("interaction_score"),
                    approved=row.get("approved"),
                    immunotherapy=row.get("immunotherapy"),
                    anti_neoplastic=row.get("anti_neoplastic"),
                )

    # PharmGKB relationships
    for _, row in relationships.iterrows():
        e1_type = row.get("Entity1_type")
        e2_type = row.get("Entity2_type")
        e1_name = norm(row.get("Entity1_name"))
        e2_name = norm(row.get("Entity2_name"))
        e1_id = row.get("Entity1_id")
        e2_id = row.get("Entity2_id")
        pmids = split_pmids(row.get("PMIDs"))

        common_kwargs = {
            "evidence_type": split_drug_tokens(row.get("Evidence")),
            "association": [row.get("Association")],
            "pmids": pmids,
            "pmid_count_numeric": len(pmids),
            "pk_related": pd.notna(row.get("PK")),
            "pd_related": pd.notna(row.get("PD")),
            "source_database": ["PharmGKB_relationships"],
        }

        if e1_type == "Gene" and e2_type == "Chemical":
            add_support(gene_drug, f"{e1_name}|||{e2_name}", source_raw_ids=[e1_id], target_raw_ids=[e2_id], **common_kwargs)
        elif e1_type == "Chemical" and e2_type == "Gene":
            add_support(gene_drug, f"{e2_name}|||{e1_name}", source_raw_ids=[e2_id], target_raw_ids=[e1_id], **common_kwargs)
        elif e1_type == "Gene" and e2_type == "Disease":
            add_support(gene_disease, f"{e1_name}|||{norm(e2_id)}", source_raw_ids=[e1_id], target_raw_ids=[e2_id], **common_kwargs)
        elif e1_type == "Disease" and e2_type == "Gene":
            add_support(gene_disease, f"{e2_name}|||{norm(e1_id)}", source_raw_ids=[e2_id], target_raw_ids=[e1_id], **common_kwargs)
        elif e1_type == "Variant" and e2_type == "Gene":
            add_support(variant_gene, f"{e1_name}|||{e2_name}", source_raw_ids=[e1_id], target_raw_ids=[e2_id], **common_kwargs)
        elif e1_type == "Gene" and e2_type == "Variant":
            add_support(variant_gene, f"{e2_name}|||{e1_name}", source_raw_ids=[e2_id], target_raw_ids=[e1_id], **common_kwargs)
        elif e1_type == "Variant" and e2_type == "Chemical":
            add_support(variant_drug, f"{e1_name}|||{e2_name}", source_raw_ids=[e1_id], target_raw_ids=[e2_id], **common_kwargs)
        elif e1_type == "Chemical" and e2_type == "Variant":
            add_support(variant_drug, f"{e2_name}|||{e1_name}", source_raw_ids=[e2_id], target_raw_ids=[e1_id], **common_kwargs)

    # clinicalVariants.tsv
    for _, row in clinical.iterrows():
        variants = [norm(v) for v in split_variant_tokens(row.get("variant")) if norm(v)]
        drugs = [norm(d) for d in split_drug_tokens(row.get("chemicals")) if norm(d)]
        gene = norm(row.get("gene"))
        phenotypes = split_drug_tokens(row.get("phenotypes"))
        evidence_level = row.get("level of evidence")
        annotation_type = row.get("type")

        for variant in variants:
            if gene:
                add_support(
                    variant_gene,
                    f"{variant}|||{gene}",
                    phenotype_names=phenotypes,
                    evidence_level=[evidence_level],
                    clinical_annotation_type=[annotation_type],
                    source_database=["clinicalVariants"],
                    pk_related="pk" in str(annotation_type).lower() or "metabolism" in str(annotation_type).lower(),
                    pd_related="pd" in str(annotation_type).lower() or "toxicity" in str(annotation_type).lower() or "efficacy" in str(annotation_type).lower(),
                )
            for drug in drugs:
                add_support(
                    variant_drug,
                    f"{variant}|||{drug}",
                    phenotype_names=phenotypes,
                    evidence_level=[evidence_level],
                    clinical_annotation_type=[annotation_type],
                    source_database=["clinicalVariants"],
                    pk_related="pk" in str(annotation_type).lower() or "metabolism" in str(annotation_type).lower(),
                    pd_related="pd" in str(annotation_type).lower() or "toxicity" in str(annotation_type).lower() or "efficacy" in str(annotation_type).lower(),
                )

    # summary_annotations.tsv
    for _, row in summary.iterrows():
        variants = [norm(v) for v in split_variant_tokens(row.get("Variant/Haplotypes")) if norm(v)]
        drugs = [norm(d) for d in split_drug_tokens(row.get("Drug(s)")) if norm(d)]
        gene = norm(row.get("Gene"))
        phenotypes = split_drug_tokens(row.get("Phenotype(s)"))
        evidence_level = row.get("Level of Evidence")
        ann_type = row.get("Phenotype Category")
        pmid_count = row.get("PMID Count")
        source_id = row.get("Summary Annotation ID")

        for variant in variants:
            if gene:
                add_support(
                    variant_gene,
                    f"{variant}|||{gene}",
                    source_raw_ids=[source_id],
                    phenotype_names=phenotypes,
                    evidence_level=[evidence_level],
                    clinical_annotation_type=[ann_type],
                    source_database=["summary_annotations"],
                    pmid_count_numeric=pmid_count,
                    pk_related="pk" in str(ann_type).lower(),
                    pd_related=any(x in str(ann_type).lower() for x in ["toxicity", "efficacy", "response", "dosage"]),
                )
            for drug in drugs:
                add_support(
                    variant_drug,
                    f"{variant}|||{drug}",
                    source_raw_ids=[source_id],
                    phenotype_names=phenotypes,
                    evidence_level=[evidence_level],
                    clinical_annotation_type=[ann_type],
                    source_database=["summary_annotations"],
                    pmid_count_numeric=pmid_count,
                    pk_related="pk" in str(ann_type).lower(),
                    pd_related=any(x in str(ann_type).lower() for x in ["toxicity", "efficacy", "response", "dosage"]),
                )

    return gene_drug, gene_disease, variant_gene, variant_drug


# -----------------------------
# Enrich nodes
# -----------------------------
def enrich_nodes(nodes, edges, gene_lookup, drug_lookup, disease_lookup, variant_lookup):
    out = nodes.copy()
    out["node_key"] = out["node_id"].astype(str).str.split(":", n=1).str[1].map(norm)

    connected = pd.concat(
        [
            edges[["source_id", "relation_type"]].rename(columns={"source_id": "node_id"}),
            edges[["target_id", "relation_type"]].rename(columns={"target_id": "node_id"}),
        ],
        ignore_index=True,
    )
    connected_map = connected.groupby("node_id")["relation_type"].apply(lambda s: join_unique(sorted(set(s.dropna())))).to_dict()

    extra_cols = [
        "node_name",
        "gene_symbol",
        "gene_name",
        "gene_family",
        "primary_gene_family",
        "gene_family_id",
        "parent_gene_family_ids",
        "parent_gene_families",
        "variant_name",
        "drug_name",
        "disease_name",
        "connected_edge_types",
    ]
    for col in extra_cols:
        out[col] = pd.Series([np.nan] * len(out), dtype="object")

    out["connected_edge_types"] = out["node_id"].map(connected_map)

    gene_mask = out["node_type"].eq("gene")
    drug_mask = out["node_type"].eq("drug")
    disease_mask = out["node_type"].eq("disease")
    variant_mask = out["node_type"].eq("variant")

    gene_cols = [
        "node_name",
        "gene_symbol",
        "gene_name",
        "gene_family",
        "primary_gene_family",
        "gene_family_id",
        "parent_gene_family_ids",
        "parent_gene_families",
    ]
    for col in gene_cols + ["gene_family_match_source"]:
        mapping = {k: v.get(col) for k, v in gene_lookup.items()}
        if col not in out.columns:
            out[col] = pd.Series([np.nan] * len(out), dtype="object")
        out.loc[gene_mask, col] = out.loc[gene_mask, "node_key"].map(mapping)
    out.loc[gene_mask & out["node_name"].isna(), "node_name"] = out.loc[gene_mask & out["node_name"].isna(), "node_key"]

    for col in ["node_name", "drug_name"]:
        key = "node_name" if col == "node_name" else "drug_name"
        mapping = {k: v.get(key) for k, v in drug_lookup.items()}
        out.loc[drug_mask, col] = out.loc[drug_mask, "node_key"].map(mapping)
    out.loc[drug_mask & out["node_name"].isna(), "node_name"] = out.loc[drug_mask & out["node_name"].isna(), "node_key"]
    out.loc[drug_mask & out["drug_name"].isna(), "drug_name"] = out.loc[drug_mask & out["drug_name"].isna(), "node_key"]

    for col in ["node_name", "disease_name"]:
        key = "node_name" if col == "node_name" else "disease_name"
        mapping = {k: v.get(key) for k, v in disease_lookup.items()}
        out.loc[disease_mask, col] = out.loc[disease_mask, "node_key"].map(mapping)
    out.loc[disease_mask & out["node_name"].isna(), "node_name"] = out.loc[disease_mask & out["node_name"].isna(), "node_key"]
    out.loc[disease_mask & out["disease_name"].isna(), "disease_name"] = out.loc[disease_mask & out["disease_name"].isna(), "node_key"]

    for col in ["node_name", "variant_name"]:
        key = "node_name" if col == "node_name" else "variant_name"
        mapping = {k: v.get(key) for k, v in variant_lookup.items()}
        out.loc[variant_mask, col] = out.loc[variant_mask, "node_key"].map(mapping)
    out.loc[variant_mask & out["node_name"].isna(), "node_name"] = out.loc[variant_mask & out["node_name"].isna(), "node_key"]
    out.loc[variant_mask & out["variant_name"].isna(), "variant_name"] = out.loc[variant_mask & out["variant_name"].isna(), "node_key"]

    out = out.drop(columns=["node_key"])
    return out

# -----------------------------
# Enrich edges
# -----------------------------
def summarize_support(record):
    if record is None:
        return {
            "source_raw_ids": np.nan,
            "target_raw_ids": np.nan,
            "phenotype_names": np.nan,
            "evidence_level": np.nan,
            "evidence_type": np.nan,
            "association": np.nan,
            "clinical_annotation_type": np.nan,
            "interaction_type": np.nan,
            "interaction_score_max": np.nan,
            "interaction_score_mean": np.nan,
            "source_database": np.nan,
            "source_version": np.nan,
            "pmids": np.nan,
            "pmid_count": np.nan,
            "pk_related": False,
            "pd_related": False,
            "approved": False,
            "immunotherapy": False,
            "anti_neoplastic": False,
            "row_count": 0,
        }

    interaction_scores = record["interaction_scores"]
    pmid_count_final = max(len(record["pmids"]), record["pmid_count_numeric"])

    return {
        "source_raw_ids": join_unique(sorted(record["source_raw_ids"])),
        "target_raw_ids": join_unique(sorted(record["target_raw_ids"])),
        "phenotype_names": join_unique(sorted(record["phenotype_names"])),
        "evidence_level": join_unique(sorted(record["evidence_level"])),
        "evidence_type": join_unique(sorted(record["evidence_type"])),
        "association": join_unique(sorted(record["association"])),
        "clinical_annotation_type": join_unique(sorted(record["clinical_annotation_type"])),
        "interaction_type": join_unique(sorted(record["interaction_type"])),
        "interaction_score_max": max(interaction_scores) if interaction_scores else np.nan,
        "interaction_score_mean": (sum(interaction_scores) / len(interaction_scores)) if interaction_scores else np.nan,
        "source_database": join_unique(sorted(record["source_database"])),
        "source_version": join_unique(sorted(record["source_version"])),
        "pmids": join_unique(sorted(record["pmids"]), sep=";"),
        "pmid_count": int(pmid_count_final) if pmid_count_final else np.nan,
        "pk_related": record["pk_related"],
        "pd_related": record["pd_related"],
        "approved": record["approved"],
        "immunotherapy": record["immunotherapy"],
        "anti_neoplastic": record["anti_neoplastic"],
        "row_count": int(record["row_count"]),
    }


def enrich_edges(edges, gene_lookup, drug_lookup, disease_lookup, variant_lookup, edge_supports):
    gene_drug_support, gene_disease_support, variant_gene_support, variant_drug_support = edge_supports

    out = edges.copy()
    out["source_key"] = out["source_raw"].map(norm)
    out["target_key"] = out["target_raw"].map(norm)
    out["pair_key"] = out["source_key"].astype(str) + "|||" + out["target_key"].astype(str)

    out["edge_key"] = out["relation_type"].astype(str) + "|" + out["source_id"].astype(str) + "|" + out["target_id"].astype(str)
    out["edge_type"] = out["relation_type"]
    out["source_node_id"] = out["source_id"]
    out["source"] = out["source_raw"]
    out["target_node_id"] = out["target_id"]
    out["target"] = out["target_raw"]

    requested_cols = [
        "source_raw_ids",
        "target_raw_ids",
        "gene_symbol",
        "gene_name",
        "gene_family",
        "primary_gene_family",
        "gene_family_id",
        "parent_gene_family_ids",
        "parent_gene_families",
        "gene_family_match_source",
        "drug_name",
        "variant_name",
        "disease_name",
        "phenotype_names",
        "evidence_level",
        "evidence_type",
        "association",
        "clinical_annotation_type",
        "interaction_type",
        "interaction_score_max",
        "interaction_score_mean",
        "source_database",
        "source_version",
        "pmids",
        "pmid_count",
        "pk_related",
        "pd_related",
        "approved",
        "immunotherapy",
        "anti_neoplastic",
        "row_count",
        "has_gene_family",
        "has_phenotype",
        "phenotype_count",
        "drug_count_context",
        "drug_class",
    ]
    for col in requested_cols:
        out[col] = pd.Series([np.nan] * len(out), dtype="object")

    support_maps = {
        "gene_drug": gene_drug_support,
        "gene_disease": gene_disease_support,
        "variant_gene": variant_gene_support,
        "variant_drug": variant_drug_support,
    }

    support_summary_cols = [
        "source_raw_ids",
        "target_raw_ids",
        "phenotype_names",
        "evidence_level",
        "evidence_type",
        "association",
        "clinical_annotation_type",
        "interaction_type",
        "interaction_score_max",
        "interaction_score_mean",
        "source_database",
        "source_version",
        "pmids",
        "pmid_count",
        "pk_related",
        "pd_related",
        "approved",
        "immunotherapy",
        "anti_neoplastic",
        "row_count",
    ]

    for relation_type, support_map in support_maps.items():
        mask = out["relation_type"].eq(relation_type)
        if not mask.any():
            continue
        needed_keys = set(out.loc[mask, "pair_key"].dropna())
        summarized = {k: summarize_support(v) for k, v in support_map.items() if k in needed_keys}
        for col in support_summary_cols:
            mapping = {k: v[col] for k, v in summarized.items()}
            out.loc[mask, col] = out.loc[mask, "pair_key"].map(mapping)

    out["gene_key"] = np.where(out["source_type"].eq("gene"), out["source_key"], np.where(out["target_type"].eq("gene"), out["target_key"], np.nan))
    out["drug_key"] = np.where(out["source_type"].eq("drug"), out["source_key"], np.where(out["target_type"].eq("drug"), out["target_key"], np.nan))
    out["variant_key"] = np.where(out["source_type"].eq("variant"), out["source_key"], np.where(out["target_type"].eq("variant"), out["target_key"], np.nan))
    out["disease_key"] = np.where(out["source_type"].eq("disease"), out["source_key"], np.where(out["target_type"].eq("disease"), out["target_key"], np.nan))

    for col in [
        "gene_symbol",
        "gene_name",
        "gene_family",
        "primary_gene_family",
        "gene_family_id",
        "parent_gene_family_ids",
        "parent_gene_families",
        "gene_family_match_source",
    ]:
        mapping = {k: v.get(col) for k, v in gene_lookup.items()}
        out[col] = out["gene_key"].map(mapping)

    drug_name_map = {k: v.get("drug_name") for k, v in drug_lookup.items()}
    drug_raw_id_map = {k: v.get("drug_raw_ids") for k, v in drug_lookup.items()}
    out["drug_name"] = out["drug_key"].map(drug_name_map)
    out.loc[out["drug_name"].isna() & out["drug_key"].notna(), "drug_name"] = out.loc[out["drug_name"].isna() & out["drug_key"].notna(), "drug_key"]
    out.loc[out["target_raw_ids"].isna() & out["drug_key"].notna(), "target_raw_ids"] = out.loc[out["target_raw_ids"].isna() & out["drug_key"].notna(), "drug_key"].map(drug_raw_id_map)

    variant_name_map = {k: v.get("variant_name") for k, v in variant_lookup.items()}
    out["variant_name"] = out["variant_key"].map(variant_name_map)
    out.loc[out["variant_name"].isna() & out["variant_key"].notna(), "variant_name"] = out.loc[out["variant_name"].isna() & out["variant_key"].notna(), "variant_key"]

    disease_name_map = {k: v.get("disease_name") for k, v in disease_lookup.items()}
    out["disease_name"] = out["disease_key"].map(disease_name_map)
    out.loc[out["disease_name"].isna() & out["disease_key"].notna(), "disease_name"] = out.loc[out["disease_name"].isna() & out["disease_key"].notna(), "disease_key"]

    out["has_gene_family"] = out["gene_family"].notna()
    out["has_phenotype"] = out["phenotype_names"].notna()
    out["phenotype_count"] = out["phenotype_names"].apply(lambda x: len(str(x).split("|")) if pd.notna(x) else 0)
    out["drug_class"] = np.nan

    context_key = np.where(out["gene_symbol"].notna(), out["gene_symbol"].astype(str), np.where(out["variant_name"].notna(), out["variant_name"].astype(str), np.nan))
    temp = pd.DataFrame({"context_key": context_key, "drug_name": out["drug_name"]})
    drug_count_map = temp[temp["drug_name"].notna()].groupby("context_key")["drug_name"].nunique().to_dict()
    out["drug_count_context"] = [drug_count_map.get(k, np.nan) if isinstance(k, str) and k != "nan" else np.nan for k in context_key]

    # Fill missing boolean / numeric defaults for easier filtering
    for col in ["pk_related", "pd_related", "approved", "immunotherapy", "anti_neoplastic"]:
        out[col] = out[col].fillna(False).astype(bool)
    out["row_count"] = out["row_count"].fillna(0).astype(int)
    out["phenotype_count"] = out["phenotype_count"].fillna(0).astype(int)

    out = out.drop(columns=["source_key", "target_key", "pair_key", "gene_key", "drug_key", "variant_key", "disease_key"])
    return out

# -----------------------------
# Main
# -----------------------------
def main(base_dir=BASE_DIR):
    nodes, edges, clinical, hgnc, hierarchy, interactions, relationships, summary = read_inputs(base_dir)

    family_name_by_id, get_ancestors = build_family_maps(hgnc, hierarchy)
    gene_lookup = build_gene_lookup(hgnc, family_name_by_id, get_ancestors)
    drug_lookup = build_drug_lookup(interactions, relationships, clinical, summary)
    disease_lookup = build_disease_lookup(relationships)
    variant_lookup = build_variant_lookup(clinical, summary, relationships)
    edge_supports = build_edge_support(interactions, relationships, clinical, summary)

    full_nodes = enrich_nodes(nodes, edges, gene_lookup, drug_lookup, disease_lookup, variant_lookup)
    full_edges = enrich_edges(edges, gene_lookup, drug_lookup, disease_lookup, variant_lookup, edge_supports)

    node_path = os.path.join(base_dir, OUTPUT_FILES["nodes"])
    edge_path = os.path.join(base_dir, OUTPUT_FILES["edges"])

    full_nodes.to_csv(node_path, index=False)
    full_edges.to_csv(edge_path, index=False)

    print("Saved:")
    print(f"- {node_path}: {full_nodes.shape}")
    print(f"- {edge_path}: {full_edges.shape}")
    print("\nRow count check:")
    print(f"Total nodes: {len(full_nodes)}")
    print(f"Total edges: {len(full_edges)}")


if __name__ == "__main__":
    main()
