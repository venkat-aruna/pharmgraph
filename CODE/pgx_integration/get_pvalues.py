#!/usr/bin/env python
"""
=============================================================================
P-Value & Significance Enrichment Script

Takes pharmgkb_frequencies_wide.tsv (or long) and adds statistical 
significance data from two sources:
    GWAS Catalog (ebi.ac.uk/gwas)
        returns p_value, beta, odds ratiom ci_lower/upper, trait, 
        study_accession and pubmed_id for each rsID
    PharmGKB Variant Annotations (var_drug_ann.tsv) 
        optional 

Can resume after cancelling; checkpoints save after every variant

Usage: 
    python get_pvalues.py

Outputs:
  gwas_pvalues.tsv         
  pharmgkb_wide_enriched.tsv 
  pharmgkb_long_enriched.tsv 

=============================================================================
"""

import pandas as pd
import requests
import time
import json
import os
import sys
import re

# =============================================================================
# CONFIGURATION — change input filenames here if yours differ
# =============================================================================

INPUT_WIDE       = "./pharmgkb_frequencies_wide.tsv"
INPUT_LONG       = "./pharmgkb_frequencies_long.tsv"
VAR_ANN_FILE     = "./var_drug_ann.tsv"           # optional extra source

OUTPUT_GWAS      = "./gwas_pvalues.tsv"            # raw GWAS Catalog hits
OUTPUT_WIDE_ENR  = "./pharmgkb_wide_enriched.tsv"  # wide + p-values
OUTPUT_LONG_ENR  = "./pharmgkb_long_enriched.tsv"  # long + p-values

CHECKPOINT_FILE  = "./checkpoint_pvalues.json"     # resume file

SLEEP_GWAS       = 0.15     # seconds between GWAS Catalog calls (limit: 7/sec)
REQUEST_TIMEOUT  = 15       # seconds before giving up on one call
MAX_RETRIES      = 3        # retry count on failure

# GWAS genome-wide significance threshold.
# p < 5e-8 is the conventional cutoff when testing millions of variants.
# Associations above this threshold may still be real but need replication.
GWAS_SIG_THRESHOLD = 5e-8

# =============================================================================
# DIAGNOSTIC MODE
# =============================================================================

def run_diagnostic():
    print("\n" + "="*60)
    print("  DIAGNOSTIC — testing GWAS Catalog API")
    print("="*60)

    test_rsids = ['rs4149056', 'rs1045642', 'rs1799853', 'rs9923231']
    # rs4149056  = SLCO1B1 statin myopathy — should have hits
    # rs1045642  = ABCB1 — very common, many studies
    # rs1799853  = CYP2C9*2 warfarin — classic PGx
    # rs9923231  = VKORC1 warfarin — extremely well studied

    for rsid in test_rsids:
        print(f"\n  Testing {rsid}...", end=' ')
        sys.stdout.flush()
        results = fetch_gwas_catalog(rsid)
        if results:
            best_p = min(r['p_value'] for r in results if r['p_value'] is not None)
            print(f"✓  {len(results)} associations found  |  best p = {best_p:.2e}")
            for r in results[:3]:
                print(f"      p={r['p_value']:.2e}  OR={r['odds_ratio']}  "
                      f"beta={r['beta']}  trait='{r['trait'][:60]}'")
        else:
            print("✗  No data")
        time.sleep(1.0)

    print("\n  If you see ✓ for at least one rsID, the API is working.")
    print("  Run without --diagnose to process your full dataset.\n")


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        print(f"  Resumed checkpoint: {len(data)} rsIDs already fetched")
        return data
    print("  No checkpoint found — starting fresh")
    return {}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# GWAS CATALOG API
#
# The GWAS Catalog REST API is at: https://www.ebi.ac.uk/gwas/rest/api
# Documentation: https://www.ebi.ac.uk/gwas/docs/api
#
# We use the /singleNucleotidePolymorphisms/{rsid}/associations endpoint.
# This returns all GWAS associations ever recorded for a variant.
# =============================================================================

GWAS_BASE = "https://www.ebi.ac.uk/gwas/rest/api"

def fetch_gwas_catalog(rsid):
    """
    Queries GWAS Catalog for all associations recorded for an rsID.

    Returns a list of dicts, one per association (study × trait combo):
    [
      {
        'rsid':            'rs4149056',
        'trait':           'Simvastatin-induced myopathy',
        'p_value':         3.2e-9,
        'beta':            0.42,
        'odds_ratio':      None,
        'ci_lower':        0.28,
        'ci_upper':        0.56,
        'n_cases':         120,
        'n_samples':       850,
        'study_accession': 'GCST000456',
        'pubmed_id':       '18987363',
        'gwas_significant': True,
      },
      ...
    ]
    or [] if the rsID has no GWAS Catalog entries.
    """
    url = f"{GWAS_BASE}/singleNucleotidePolymorphisms/{rsid}/associations"
    # This endpoint returns all associations for a given rsID.
    # It's paginated — we request page_size=200 to minimise requests.

    params = {
        'projection': 'associationBySnp',
        # 'associationBySnp' projection gives us the full association details
        # including p-values, betas, odds ratios, and linked study info.
        'size': 200,
        'page': 0,
    }

    all_results = []

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT
            )

            if r.status_code == 404:
                return []
                # 404 = this rsID has no GWAS Catalog entries at all. Normal.

            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue

            if r.status_code != 200:
                return []

            data = r.json()

            # Navigate the HAL JSON structure the GWAS Catalog uses.
            # Response looks like:
            # {
            #   "_embedded": {
            #     "associations": [ {...}, {...}, ... ]
            #   },
            #   "page": {
            #     "totalElements": 5,
            #     "totalPages": 1
            #   }
            # }
            embedded = data.get('_embedded', {})
            associations = embedded.get('associations', [])

            for assoc in associations:
                parsed = parse_gwas_association(rsid, assoc)
                if parsed:
                    all_results.append(parsed)

            # Check if there are more pages
            page_info = data.get('page', {})
            total_pages = page_info.get('totalPages', 1)
            current_page = page_info.get('number', 0)

            if current_page + 1 < total_pages:
                # There are more pages — fetch them
                for page_num in range(1, total_pages):
                    params['page'] = page_num
                    r2 = requests.get(
                        url,
                        params=params,
                        headers={"Accept": "application/json"},
                        timeout=REQUEST_TIMEOUT
                    )
                    if r2.status_code == 200:
                        data2 = r2.json()
                        for assoc in data2.get('_embedded', {}).get('associations', []):
                            parsed = parse_gwas_association(rsid, assoc)
                            if parsed:
                                all_results.append(parsed)
                    time.sleep(SLEEP_GWAS)

            return all_results

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue
        except Exception:
            return []

    return []


def parse_gwas_association(rsid, assoc):
    """
    Parses one association record from the GWAS Catalog response.
    Returns a clean dict or None if the record is malformed.

    The raw GWAS Catalog association object has a complex nested structure.
    This function flattens it into something useful.
    """
    try:
        # --- P-value ---
        # GWAS Catalog stores p-value as mantissa × 10^exponent
        # e.g. mantissa=3.2, exponent=-9 → p = 3.2e-9
        mantissa = assoc.get('pvalueMantissa')
        exponent = assoc.get('pvalueExponent')
        p_value = None
        if mantissa is not None and exponent is not None:
            try:
                p_value = float(mantissa) * (10 ** int(exponent))
                # Convert back to a proper float like 3.2e-9
            except (TypeError, ValueError):
                p_value = None

        # --- Effect sizes ---
        beta_num    = assoc.get('betaNum')
        beta_unit   = assoc.get('betaUnit', '')
        # betaNum: the numeric beta coefficient from the regression
        # betaUnit: what scale/unit the beta is in (e.g. "SD increase per allele")

        odds_ratio  = assoc.get('orPerCopyNum')
        # orPerCopyNum: odds ratio per copy of the effect allele
        # Used for binary traits (disease yes/no, adverse event yes/no)

        # --- Confidence interval ---
        # GWAS Catalog stores CI as a string like "[0.28-0.56]" or "0.28-0.56"
        ci_raw = assoc.get('range', '') or ''
        ci_lower, ci_upper = parse_confidence_interval(ci_raw)

        # --- Risk allele ---
        risk_allele = None
        loci = assoc.get('loci', [])
        if loci:
            for locus in loci:
                for ra in locus.get('strongestRiskAlleles', []):
                    ra_name = ra.get('riskAlleleName', '')
                    # riskAlleleName looks like "rs4149056-C"
                    if '-' in ra_name:
                        risk_allele = ra_name.split('-')[-1]
                        break

        # --- Trait ---
        trait = None
        efo_traits = assoc.get('efoTraits', [])
        if efo_traits:
            trait = efo_traits[0].get('trait', None)
            # efoTraits: Experimental Factor Ontology trait labels
            # More standardized than free-text descriptions

        # --- Study info ---
        study = assoc.get('study', {}) or {}
        pubmed_id       = study.get('publicationInfo', {}).get('pubmedId')
        study_accession = study.get('accessionId')
        # accessionId: the GWAS Catalog's own study ID (GCST######)
        # Use this to look up the full study at ebi.ac.uk/gwas/studies/GCST######

        # --- Sample sizes ---
        n_cases   = study.get('initialSampleSize', None)
        # initialSampleSize is a text description like "120 cases, 730 controls"
        # We keep it as text since it varies in format

        # --- Trait description (fallback if efoTraits is empty) ---
        if not trait:
            trait = assoc.get('traitName', None)

        return {
            'rsid':             rsid,
            'trait':            trait or 'Unknown',
            'p_value':          p_value,
            'beta':             float(beta_num) if beta_num is not None else None,
            'beta_unit':        beta_unit or None,
            'odds_ratio':       float(odds_ratio) if odds_ratio is not None else None,
            'ci_lower':         ci_lower,
            'ci_upper':         ci_upper,
            'risk_allele':      risk_allele,
            'sample_size_desc': n_cases,
            'study_accession':  study_accession,
            'pubmed_id':        str(pubmed_id) if pubmed_id else None,
            'gwas_significant': (p_value is not None and p_value < GWAS_SIG_THRESHOLD),
            # gwas_significant = True if p < 5e-8 (genome-wide significance threshold)
        }

    except Exception:
        return None


def parse_confidence_interval(ci_str):
    """
    Parses CI strings like "[0.28-0.56]", "0.28-0.56", "[1.2, 3.4]" etc.
    Returns (lower, upper) as floats, or (None, None) if unparseable.
    """
    if not ci_str:
        return None, None
    ci_str = ci_str.strip('[]() ')
    # Try comma-separated first: "0.28, 0.56"
    if ',' in ci_str:
        parts = ci_str.split(',')
        if len(parts) == 2:
            try:
                return float(parts[0].strip()), float(parts[1].strip())
            except ValueError:
                pass
    # Try dash-separated: "0.28-0.56"
    # Be careful: negative numbers like "-0.56-0.28" also exist
    # Strategy: find the dash that separates two numbers (not a minus sign)
    matches = re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', ci_str)
    if len(matches) >= 2:
        try:
            return float(matches[0]), float(matches[1])
        except ValueError:
            pass
    return None, None


# =============================================================================
# MAIN GWAS FETCH LOOP
# =============================================================================

def fetch_all_gwas(rsids):
    """
    Fetches GWAS Catalog associations for every rsID.
    Returns a dict: rsid → list of association dicts.
    Saves progress to checkpoint after each rsID.
    """
    print("\n" + "="*60)
    print("[STEP 2] Fetching p-values from GWAS Catalog")
    print("  Source: ebi.ac.uk/gwas")
    print("="*60)

    cache = load_checkpoint()
    total = len(rsids)
    n_have = sum(1 for r in rsids if r in cache)

    print(f"\n  Total rsIDs:     {total}")
    print(f"  Already fetched: {n_have}")
    print(f"  Still to fetch:  {total - n_have}")
    print(f"\n  Press Ctrl+C at any time — progress is saved.\n")

    for i, rsid in enumerate(rsids):
        pct = round((i + 1) / total * 100)
        print(f"\r  [{i+1}/{total}] {pct}%  |  {rsid}                    ", end='')
        sys.stdout.flush()

        if rsid in cache:
            continue

        results = fetch_gwas_catalog(rsid)
        cache[rsid] = results
        # We store even empty lists [] so we don't retry rsIDs that genuinely
        # have no GWAS hits. They'd just slow down reruns.
        save_checkpoint(cache)
        time.sleep(SLEEP_GWAS)

    print()
    n_with_hits = sum(1 for v in cache.values() if v)
    n_empty     = sum(1 for v in cache.values() if not v)
    total_assoc = sum(len(v) for v in cache.values())

    print(f"\n  rsIDs with GWAS hits: {n_with_hits}")
    print(f"  rsIDs with no hits:   {n_empty}  (normal — not all PGx variants are in GWAS Catalog)")
    print(f"  Total associations:   {total_assoc}")

    return cache


# =============================================================================
# LOAD OPTIONAL PharmGKB VARIANT ANNOTATIONS (study-level p-values)
# =============================================================================

def load_pharmgkb_variant_annotations(filepath):
    if not os.path.exists(filepath):
        print(f"\n  (Optional file '{filepath}' not found — skipping PharmGKB p-values)")
        print("  To get these: pharmgkb.org/downloads → 'Variant Annotations' → var_drug_ann.tsv")
        return {}

    print(f"\n  Loading PharmGKB variant annotations: {filepath}")
    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    print(f"  Rows: {len(df)}")

    col_variant = next((c for c in df.columns if 'variant' in c.lower() or 'haplotype' in c.lower()), None)
    col_drug    = next((c for c in df.columns if 'drug' in c.lower() or 'chemical' in c.lower()), None)
    col_pval    = next((c for c in df.columns if 'p value' in c.lower() or 'pvalue' in c.lower()), None)
    col_sig     = next((c for c in df.columns if c.lower() == 'significance'), None)
    col_pmid    = next((c for c in df.columns if c.lower() == 'pmid'), None)
    col_sentence= next((c for c in df.columns if 'sentence' in c.lower()), None)
    col_gene    = next((c for c in df.columns if c.lower() == 'gene'), None)

    print(f"  Columns found: pval='{col_pval}', sig='{col_sig}', pmid='{col_pmid}'")

    ann_map = {}   # rsid → list of study records
    for _, row in df.iterrows():
        if not col_variant:
            continue
        variant_str = str(row.get(col_variant, ''))
        rsids = re.findall(r'rs\d+', variant_str)
        for rsid in rsids:
            if rsid not in ann_map:
                ann_map[rsid] = []

            p_text = str(row[col_pval]).strip() if col_pval and pd.notna(row.get(col_pval)) else None
            p_numeric = parse_pvalue_text(p_text)
            # parse_pvalue_text converts strings like "<0.05", "0.001", "NS" to floats

            ann_map[rsid].append({
                'drug':        str(row[col_drug])    if col_drug    and pd.notna(row.get(col_drug))    else None,
                'gene':        str(row[col_gene])    if col_gene    and pd.notna(row.get(col_gene))    else None,
                'p_value_text':    p_text,
                'p_value_numeric': p_numeric,
                # p_value_numeric: our best attempt to convert the text to a float.
                # None if it was something like "NS" or unparseable.
                'significant': str(row[col_sig])     if col_sig     and pd.notna(row.get(col_sig))     else None,
                'pmid':        str(row[col_pmid])    if col_pmid    and pd.notna(row.get(col_pmid))    else None,
                'sentence':    str(row[col_sentence])if col_sentence and pd.notna(row.get(col_sentence))else None,
            })

    print(f"  rsIDs with variant annotation p-values: {len(ann_map)}")
    return ann_map


def parse_pvalue_text(text):
    """
    Tries to extract a numeric p-value from messy text like:
      "0.001"         → 0.001
      "< 0.05"        → 0.05   (conservative: use the threshold value)
      "1.2 x 10-8"   → 1.2e-8
      "1.2E-8"        → 1.2e-8
      "NS"            → None   (not significant, no numeric value)
      "not significant" → None
    """
    if not text or text in ('None', 'nan', 'NS', 'not significant', 'n.s.', 'N/A'):
        return None
    text = text.strip().lower()
    if 'not significant' in text or text == 'ns':
        return None
    # Remove comparison operators
    text = re.sub(r'[<>=≤≥]', '', text).strip()
    # Normalise scientific notation: "1.2 x 10-8" → "1.2e-8"
    text = re.sub(r'\s*[×x]\s*10\s*[\^]?\s*(-?\d+)', r'e\1', text)
    try:
        return float(text)
    except ValueError:
        # Try extracting just the first number
        match = re.search(r'\d+\.?\d*(?:e[+-]?\d+)?', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None


# =============================================================================
# BUILD SUMMARY STATS PER rsID
# =============================================================================

def build_summary_per_rsid(gwas_cache, var_ann_map):
    """
    For each rsID, collapses all GWAS hits into a single summary row with:
      - best_p_value          smallest p-value seen across all studies
      - n_gwas_studies        how many GWAS studies found this variant
      - n_gwas_significant    how many studies met genome-wide significance
      - best_odds_ratio       OR from the most significant study
      - best_beta             beta from the most significant study
      - best_ci_lower/upper   CI from the most significant study
      - best_trait            trait from the most significant study
      - best_study            study accession from the most significant study
      - best_pubmed_id        PubMed ID from the most significant study
      - pharmgkb_best_p       best numeric p-value from PharmGKB variant annotations
      - pharmgkb_sig_count    number of "significant=yes" records in PharmGKB

    This summary is what we join onto your wide/long TSV.
    """
    summaries = {}

    all_rsids = set(gwas_cache.keys()) | set(var_ann_map.keys())

    for rsid in all_rsids:
        gwas_hits = gwas_cache.get(rsid, [])
        pharmgkb_hits = var_ann_map.get(rsid, [])

        # --- GWAS Catalog summary ---
        valid_hits = [h for h in gwas_hits if h.get('p_value') is not None]

        best_p         = None
        best_hit       = None
        n_sig          = 0

        if valid_hits:
            best_hit = min(valid_hits, key=lambda h: h['p_value'])
            best_p   = best_hit['p_value']
            n_sig    = sum(1 for h in valid_hits if h.get('gwas_significant'))

        # --- PharmGKB Variant Annotations summary ---
        pgkb_pvals = [
            h['p_value_numeric']
            for h in pharmgkb_hits
            if h.get('p_value_numeric') is not None
        ]
        pgkb_best_p = min(pgkb_pvals) if pgkb_pvals else None

        pgkb_sig_count = sum(
            1 for h in pharmgkb_hits
            if str(h.get('significant', '')).lower() == 'yes'
        )

        summaries[rsid] = {
            # GWAS Catalog fields
            'best_gwas_p_value':    best_p,
            # The smallest p-value from any GWAS study for this variant.
            # This is your primary significance measure.

            'gwas_significant':     (best_p is not None and best_p < GWAS_SIG_THRESHOLD),
            # True = at least one study found p < 5e-8 (genome-wide significant).

            'n_gwas_studies':       len(gwas_hits),
            # Total number of GWAS associations recorded (all traits, all studies).

            'n_gwas_significant':   n_sig,
            # Number of those associations that met genome-wide significance.

            'best_gwas_odds_ratio': best_hit.get('odds_ratio')    if best_hit else None,
            # Odds ratio from the most significant study.
            # Applies to binary outcomes (adverse event: yes/no).

            'best_gwas_beta':       best_hit.get('beta')          if best_hit else None,
            # Beta coefficient from the most significant study.
            # Applies to continuous outcomes (plasma level, enzyme activity).

            'best_gwas_ci_lower':   best_hit.get('ci_lower')      if best_hit else None,
            'best_gwas_ci_upper':   best_hit.get('ci_upper')      if best_hit else None,
            # 95% confidence interval. If CI doesn't cross 0 (beta) or 1 (OR) = significant.

            'best_gwas_trait':      best_hit.get('trait')         if best_hit else None,
            # What phenotype/trait was studied in the most significant GWAS.

            'best_gwas_study':      best_hit.get('study_accession') if best_hit else None,
            # GWAS Catalog study ID. Look it up at ebi.ac.uk/gwas/studies/{id}

            'best_gwas_pubmed_id':  best_hit.get('pubmed_id')     if best_hit else None,
            # PubMed ID of the source paper. Look it up at pubmed.ncbi.nlm.nih.gov/{id}

            # PharmGKB Variant Annotations fields (only if you have var_drug_ann.tsv)
            'pharmgkb_best_p_value': pgkb_best_p,
            # Best numeric p-value extracted from PharmGKB variant annotations.
            # More drug-specific than GWAS Catalog.

            'pharmgkb_sig_studies':  pgkb_sig_count,
            # Number of papers PharmGKB curators marked as "significant=yes".
        }

    return summaries


# =============================================================================
# LOAD INPUT TSVs
# =============================================================================

def load_input_tsvs():
    print("\n" + "="*60)
    print("[STEP 1] Loading your existing TSV files")
    print("="*60)

    wide_df = None
    long_df = None

    if os.path.exists(INPUT_WIDE):
        wide_df = pd.read_csv(INPUT_WIDE, sep='\t', low_memory=False)
        print(f"\n  Wide TSV: {len(wide_df)} rows × {len(wide_df.columns)} columns")
        print(f"  Columns: {list(wide_df.columns)}")
    else:
        print(f"\n  WARNING: Wide TSV not found at {INPUT_WIDE}")

    if os.path.exists(INPUT_LONG):
        long_df = pd.read_csv(INPUT_LONG, sep='\t', low_memory=False)
        print(f"\n  Long TSV: {len(long_df)} rows × {len(long_df.columns)} columns")
    else:
        print(f"\n  WARNING: Long TSV not found at {INPUT_LONG}")

    if wide_df is None and long_df is None:
        print("\n  ERROR: Neither input file found. Run gnomad_variant_freq.py first.")
        sys.exit(1)

    # Extract all unique rsIDs from whichever file(s) we have
    all_rsids = set()
    if wide_df is not None and 'rsid' in wide_df.columns:
        all_rsids.update(wide_df['rsid'].dropna().unique())
    if long_df is not None and 'rsid' in long_df.columns:
        all_rsids.update(long_df['rsid'].dropna().unique())

    print(f"\n  Total unique rsIDs to look up: {len(all_rsids)}")
    return wide_df, long_df, sorted(all_rsids)


# =============================================================================
# SAVE RAW GWAS HITS FILE
# =============================================================================

def save_gwas_raw(gwas_cache):
    """
    Saves all raw GWAS Catalog associations to gwas_pvalues.tsv.
    One row per association (variant × study × trait).
    This is the full detail file — use it to dig into specific variants.
    """
    rows = []
    for rsid, hits in gwas_cache.items():
        for hit in hits:
            rows.append(hit)

    if not rows:
        print("  No GWAS hits to save.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by p_value ascending (most significant first)
    if 'p_value' in df.columns:
        df = df.sort_values('p_value', na_position='last')

    df.to_csv(OUTPUT_GWAS, sep='\t', index=False)
    print(f"  ✓ Raw GWAS hits: {OUTPUT_GWAS}  ({len(df)} rows)")
    return df


# =============================================================================
# ENRICH AND SAVE WIDE/LONG TSVs
# =============================================================================

def enrich_and_save(wide_df, long_df, summaries):
    """
    Joins the per-rsID summary stats onto both the wide and long TSVs.
    The join key is 'rsid'.
    """
    summary_df = pd.DataFrame.from_dict(summaries, orient='index')
    # orient='index' means: the dict keys (rsIDs) become the index,
    # dict values (the summary dicts) become the columns.
    summary_df.index.name = 'rsid'
    summary_df = summary_df.reset_index()
    # reset_index() moves 'rsid' from index back into a regular column
    # so we can merge on it.

    sig_cols = [
        'best_gwas_p_value', 'gwas_significant', 'n_gwas_studies',
        'n_gwas_significant', 'best_gwas_odds_ratio', 'best_gwas_beta',
        'best_gwas_ci_lower', 'best_gwas_ci_upper', 'best_gwas_trait',
        'best_gwas_study', 'best_gwas_pubmed_id',
        'pharmgkb_best_p_value', 'pharmgkb_sig_studies',
    ]
    # Only keep columns that actually exist in summary_df
    sig_cols = [c for c in sig_cols if c in summary_df.columns]

    print()

    # --- Enrich wide TSV ---
    if wide_df is not None and 'rsid' in wide_df.columns:
        wide_enriched = wide_df.merge(
            summary_df[['rsid'] + sig_cols],
            on='rsid',
            how='left'
            # how='left': keep ALL rows from wide_df, add summary columns where available.
            # Variants with no GWAS hits get NaN in the significance columns.
        )
        wide_enriched.to_csv(OUTPUT_WIDE_ENR, sep='\t', index=False)
        n_with_p = wide_enriched['best_gwas_p_value'].notna().sum()
        print(f"  ✓ Wide enriched: {OUTPUT_WIDE_ENR}")
        print(f"    {len(wide_enriched)} variants  |  {n_with_p} have GWAS p-values")

    # --- Enrich long TSV ---
    if long_df is not None and 'rsid' in long_df.columns:
        long_enriched = long_df.merge(
            summary_df[['rsid'] + sig_cols],
            on='rsid',
            how='left'
        )
        long_enriched.to_csv(OUTPUT_LONG_ENR, sep='\t', index=False)
        n_with_p = long_enriched['best_gwas_p_value'].notna().sum()
        print(f"  ✓ Long enriched: {OUTPUT_LONG_ENR}")
        print(f"    {len(long_enriched)} rows  |  {n_with_p} have GWAS p-values")


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_summary(summaries, gwas_raw_df):
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    all_vals        = list(summaries.values())
    n_any_gwas      = sum(1 for v in all_vals if v['n_gwas_studies'] > 0)
    n_gwas_sig      = sum(1 for v in all_vals if v['gwas_significant'])
    n_pharmgkb_p    = sum(1 for v in all_vals if v.get('pharmgkb_best_p_value') is not None)

    print(f"\n  rsIDs processed:                {len(summaries)}")
    print(f"  Have any GWAS Catalog hits:     {n_any_gwas}")
    print(f"  Genome-wide significant (p<5e-8): {n_gwas_sig}")
    print(f"  Have PharmGKB p-values:         {n_pharmgkb_p}")

    # Top 10 most significant variants
    sig_variants = [
        (rsid, v['best_gwas_p_value'], v['best_gwas_trait'], v['n_gwas_significant'])
        for rsid, v in summaries.items()
        if v['best_gwas_p_value'] is not None
    ]
    sig_variants.sort(key=lambda x: x[1])

    if sig_variants:
        print("\n  Top 10 most significant variants (by best GWAS p-value):\n")
        print(f"  {'rsID':<15} {'p-value':<12} {'sig studies':<14} {'top trait'}")
        print("  " + "-"*75)
        for rsid, p, trait, n_sig in sig_variants[:10]:
            trait_short = (trait or 'Unknown')[:45]
            print(f"  {rsid:<15} {p:<12.2e} {n_sig:<14} {trait_short}")

    # Most replicated (most studies)
    replicated = sorted(
        [(rsid, v['n_gwas_studies'], v['best_gwas_p_value'])
         for rsid, v in summaries.items() if v['n_gwas_studies'] > 0],
        key=lambda x: -x[1]
    )
    if replicated:
        print("\n  Top 10 most studied variants (most GWAS associations):\n")
        print(f"  {'rsID':<15} {'n studies':<12} {'best p-value'}")
        print("  " + "-"*45)
        for rsid, n, p in replicated[:10]:
            p_str = f"{p:.2e}" if p is not None else "N/A"
            print(f"  {rsid:<15} {n:<12} {p_str}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    if '--diagnose' in sys.argv:
        run_diagnostic()
        sys.exit(0)

    print("="*60)
    print("  P-Value & Significance Enrichment Script")
    print("  Sources: GWAS Catalog + PharmGKB Variant Annotations")
    print("="*60)

    try:
        import pandas, requests
        print("\n✓ Libraries OK (pandas, requests)")
    except ImportError as e:
        print(f"\nERROR: {e}\nRun: pip install pandas requests")
        sys.exit(1)

    # Step 1: Load your existing wide/long TSVs and extract rsIDs
    wide_df, long_df, all_rsids = load_input_tsvs()

    # Step 2: Fetch GWAS Catalog p-values for every rsID
    gwas_cache = fetch_all_gwas(all_rsids)

    # Step 3: Optionally load PharmGKB variant annotations (study-level p-values)
    print("\n" + "="*60)
    print("[STEP 3] Loading PharmGKB variant annotations (optional)")
    print("="*60)
    var_ann_map = load_pharmgkb_variant_annotations(VAR_ANN_FILE)

    # Step 4: Build per-rsID summary (collapse all studies to best values)
    print("\n" + "="*60)
    print("[STEP 4] Building per-rsID significance summary")
    print("="*60)
    summaries = build_summary_per_rsid(gwas_cache, var_ann_map)
    print(f"  Summary built for {len(summaries)} rsIDs")

    # Step 5: Save raw GWAS hits file + enriched wide/long TSVs
    print("\n" + "="*60)
    print("[STEP 5] Saving output files")
    print("="*60)
    gwas_raw_df = save_gwas_raw(gwas_cache)
    enrich_and_save(wide_df, long_df, summaries)

    # Step 6: Print summary statistics
    print_summary(summaries, gwas_raw_df)

    print("\n" + "="*60)
    print("  DONE! Files created:")
    print(f"    {OUTPUT_GWAS}       — all raw GWAS hits (one row per study)")
    print(f"    {OUTPUT_WIDE_ENR}  — wide TSV + p-values added")
    print(f"    {OUTPUT_LONG_ENR}  — long TSV + p-values added")
    print("\n  New columns added to your TSVs:")
    print("    best_gwas_p_value      smallest p across all GWAS studies")
    print("    gwas_significant       True if p < 5e-8")
    print("    n_gwas_studies         how many GWAS studies found this variant")
    print("    n_gwas_significant     how many met genome-wide significance")
    print("    best_gwas_odds_ratio   effect size (binary outcomes)")
    print("    best_gwas_beta         effect size (continuous outcomes)")
    print("    best_gwas_ci_lower/upper  confidence interval")
    print("    best_gwas_trait        what was studied")
    print("    best_gwas_pubmed_id    the paper (pubmed.ncbi.nlm.nih.gov/{id})")
    print("    pharmgkb_best_p_value  best p from PharmGKB papers (if available)")
    print("="*60)
