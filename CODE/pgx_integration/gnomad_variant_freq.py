#!/usr/bin/env python
"""
=============================================================================
PharmGKB Variant Frequency Calculator — FIXED VERSION
=============================================================================

WHAT CHANGED FROM THE PREVIOUS VERSION:
  The gnomAD GraphQL API now blocks programmatic access (returns 403 Forbidden).
  This version uses TWO alternative sources instead:

  SOURCE 1 — Ensembl REST API /variation endpoint
    - Returns gnomAD population frequencies directly
    - Free, no account needed
    - URL: https://rest.ensembl.org/variation/human/{rsid}

  SOURCE 2 — Ensembl REST API /vep (fallback)
    - Variant Effect Predictor — annotates variants with gnomAD freqs
    - Used if source 1 has no population data for a variant

  Both return the same gnomAD population codes:
    AFR, AMR, ASJ, EAS, FIN, NFE, OTH, SAS

BEFORE YOU RUN:
  pip install pandas requests

HOW TO RUN:
  python gnomad_variant_freq.py

HOW TO RESUME AFTER CANCELLING:
  Run the same command again — checkpoints save after every variant.

RUN THE DIAGNOSTIC FIRST (optional but recommended):
  python gnomad_variant_freq.py --diagnose
  This tests your API connection and shows what data looks like before
  running the full job.
=============================================================================
"""

import pandas as pd
import requests
import time
import re
import os
import json
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

PHARMGKB_FILE     = "./summary_annotations.tsv"
OUTPUT_WIDE       = "./pharmgkb_frequencies_wide.tsv"
OUTPUT_LONG       = "./pharmgkb_frequencies_long.tsv"
OUTPUT_DRUG_MAP   = "./drug_response_map.tsv"
CHECKPOINT_COORDS = "./checkpoint_coords.json"
CHECKPOINT_FREQS  = "./checkpoint_frequencies.json"

SLEEP_API = 0.34          # ~3 requests/sec — safe for Ensembl's limit of 15/sec
REQUEST_TIMEOUT = 15      # seconds before giving up on a single API call
MAX_RETRIES = 3           # how many times to retry a failed API call

# Population codes returned by Ensembl/gnomAD and their human-readable names
POPULATION_MAP = {
    'gnomAD:afr': 'African/African American',
    'gnomAD:amr': 'Admixed American',
    'gnomAD:asj': 'Ashkenazi Jewish',
    'gnomAD:eas': 'East Asian',
    'gnomAD:fin': 'Finnish',
    'gnomAD:mid': 'Middle Eastern',
    'gnomAD:nfe': 'Non-Finnish European',
    'gnomAD:oth': 'Other',
    'gnomAD:sas': 'South Asian',
    # Ensembl sometimes uses these alternate keys for the same populations:
    'gnomADg:afr': 'African/African American',
    'gnomADg:amr': 'Admixed American',
    'gnomADg:asj': 'Ashkenazi Jewish',
    'gnomADg:eas': 'East Asian',
    'gnomADg:fin': 'Finnish',
    'gnomADg:mid': 'Middle Eastern',
    'gnomADg:nfe': 'Non-Finnish European',
    'gnomADg:oth': 'Other',
    'gnomADg:sas': 'South Asian',
    'gnomADg:ami': 'Amish',
    'gnomADg:remaining': 'Remaining/Other',
}

# Short codes for column naming (strip the "gnomAD:" prefix)
def pop_to_code(pop_key):
    return pop_key.split(':')[-1].lower()

# The clean set of output population codes we want
OUTPUT_POPS = ['afr', 'amr', 'asj', 'eas', 'fin', 'mid', 'nfe', 'oth', 'sas']


# =============================================================================
# DIAGNOSTIC MODE
# Run with: python gnomad_variant_freq.py --diagnose
# Tests API connectivity and shows what data looks like before full run
# =============================================================================

def run_diagnostic():
    print("\n" + "="*60)
    print("  DIAGNOSTIC MODE")
    print("  Testing API connection with 3 known variants")
    print("="*60)

    test_rsids = ['rs4149056', 'rs1045642', 'rs1799853']
    # rs4149056 = SLCO1B1 statin myopathy (well-known, definitely in gnomAD)
    # rs1045642 = ABCB1 (very common, present in all databases)
    # rs1799853 = CYP2C9*2 (classic pharmacogenomics variant)

    for rsid in test_rsids:
        print(f"\n  Testing {rsid}...")
        freq_dict = fetch_frequencies_from_ensembl(rsid)
        if freq_dict:
            print(f"  ✓ SUCCESS — got {len(freq_dict)} population frequencies")
            for pop, freq in sorted(freq_dict.items()):
                print(f"      {pop:<6} = {freq:.4f}")
        else:
            print(f"  ✗ FAILED — no data returned")
            print("    Possible causes:")
            print("    - No internet connection")
            print("    - Ensembl API is down (check https://rest.ensembl.org)")
            print("    - Rate limiting (try again in a few minutes)")
        time.sleep(1.0)

    print("\n  Diagnostic complete. If at least one variant succeeded,")
    print("  the script should work for the full dataset.")
    print("  Run without --diagnose to process all variants.\n")


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def load_checkpoint(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"  Resumed: {len(data)} entries from {filepath}")
        return data
    print(f"  No checkpoint at {filepath} — starting fresh")
    return {}

def save_checkpoint(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# STEP 1: LOAD PharmGKB
# =============================================================================

def load_pharmgkb(filepath):
    print("\n" + "="*60)
    print("[STEP 1] Loading PharmGKB annotations")
    print("="*60)

    if not os.path.exists(filepath):
        print(f"\n  ERROR: Cannot find: {filepath}")
        print("  Download 'Clinical Annotations' TSV from pharmgkb.org/downloads")
        return {}

    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    print(f"\n  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    col_variant  = next((c for c in df.columns if 'variant' in c.lower() or 'haplotype' in c.lower()), None)
    col_gene     = next((c for c in df.columns if c.lower() == 'gene'), None)
    col_drug     = next((c for c in df.columns if 'drug' in c.lower() or 'chemical' in c.lower()), None)
    col_evidence = next((c for c in df.columns if 'evidence' in c.lower() or 'level' in c.lower()), None)
    col_pheno    = next((c for c in df.columns if 'phenotype' in c.lower() or 'category' in c.lower()), None)

    print(f"\n  Column mapping:")
    print(f"    Variant   → '{col_variant}'")
    print(f"    Gene      → '{col_gene}'")
    print(f"    Drug      → '{col_drug}'")
    print(f"    Evidence  → '{col_evidence}'")
    print(f"    Phenotype → '{col_pheno}'")

    if col_variant is None:
        print("\n  ERROR: No variant column found.")
        return {}

    rsid_map = {}
    for _, row in df.iterrows():
        variant_str = str(row.get(col_variant, ''))
        rsids = re.findall(r'rs\d+', variant_str)
        for rsid in rsids:
            if rsid not in rsid_map:
                rsid_map[rsid] = []
            rsid_map[rsid].append({
                'gene':      str(row[col_gene])     if col_gene     and pd.notna(row[col_gene])     else 'Unknown',
                'drug':      str(row[col_drug])     if col_drug     and pd.notna(row[col_drug])     else 'Unknown',
                'evidence':  str(row[col_evidence]) if col_evidence and pd.notna(row[col_evidence]) else 'Unknown',
                'phenotype': str(row[col_pheno])    if col_pheno    and pd.notna(row[col_pheno])    else 'Unknown',
            })

    print(f"\n  Unique rsIDs: {len(rsid_map)}")
    return rsid_map


# =============================================================================
# STEP 2: FETCH FREQUENCIES FROM ENSEMBL
# =============================================================================
#
# HOW THIS WORKS:
#   Ensembl's variation endpoint returns an object for each rsID that includes
#   a 'populations' list. Each entry in that list has:
#     - population: e.g. "gnomAD:afr", "gnomAD:nfe", "1000GENOMES:CEU", etc.
#     - frequency: the allele frequency (0.0 to 1.0)
#     - allele: which allele this frequency is for ("A", "C", "T", "G")
#
#   We filter to only the gnomAD populations (entries starting with "gnomAD:")
#   and extract the alternate allele frequency (the non-reference one).
#
# WHY ENSEMBL AND NOT GNOMAD DIRECTLY?
#   gnomAD's API now returns 403 Forbidden for non-browser requests.
#   Ensembl mirrors gnomAD data and has a stable, open REST API.
# =============================================================================

def fetch_frequencies_from_ensembl(rsid):
    """
    Queries the Ensembl REST API for population frequencies for one rsID.
    Returns dict like {'afr': 0.015, 'nfe': 0.154, ...} or None on failure.

    The API endpoint used:
      GET https://rest.ensembl.org/variation/human/{rsid}?population_genotypes=1
    """
    url = f"https://rest.ensembl.org/variation/human/{rsid}?population_genotypes=1"

    for attempt in range(MAX_RETRIES):
        # MAX_RETRIES = 3, so we try up to 3 times before giving up.
        # This handles temporary network hiccups automatically.
        try:
            response = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 429:
                # 429 = Too Many Requests (rate limited)
                # Wait longer and retry
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s on successive retries
                time.sleep(wait)
                continue

            if response.status_code != 200:
                return None

            data = response.json()
            # The response looks like:
            # {
            #   "name": "rs4149056",
            #   "var_class": "SNP",
            #   "population_genotypes": [
            #     {
            #       "population": "gnomAD:afr",
            #       "genotype": "T|T",
            #       "frequency": 0.9848,
            #       "count": 5,
            #       "subpopulation": "..."
            #     },
            #     {
            #       "population": "gnomAD:afr",
            #       "genotype": "T|C",
            #       "frequency": 0.0152,   <-- this is the alt allele freq we want
            #       ...
            #     },
            #     ...
            #   ],
            #   "mappings": [...],
            #   ...
            # }

            pop_genotypes = data.get('population_genotypes', [])
            if not pop_genotypes:
                return None

            # We need to figure out which is the reference allele and which
            # is the alternate allele. Get the reference from mappings.
            ref_allele = None
            for mapping in data.get('mappings', []):
                allele_str = mapping.get('allele_string', '')
                if '/' in allele_str:
                    ref_allele = allele_str.split('/')[0]
                    break

            freq_dict = {}

            for entry in pop_genotypes:
                pop_key = entry.get('population', '')

                # Only keep gnomAD populations (skip 1000Genomes, ESP, etc.)
                if not (pop_key.startswith('gnomAD:') or pop_key.startswith('gnomADg:')):
                    continue

                # Skip sub-population breakdowns (e.g. gnomAD:afr_female)
                code = pop_to_code(pop_key)   # e.g. 'afr', 'nfe', 'afr_female'
                if '_' in code:
                    continue

                genotype = entry.get('genotype', '')
                freq = float(entry.get('frequency', 0.0))

                # Determine if this genotype entry represents the alternate allele.
                # Genotypes look like "T|C" (het) or "C|C" (hom alt) or "T|T" (hom ref).
                # We want to find the frequency of the alternate (non-reference) allele.
                # Strategy: if the genotype contains the non-reference allele, record it.
                # We look for heterozygous entries (one ref, one alt) and use their freq.
                alleles_in_genotype = set(re.split(r'[|/]', genotype))

                if ref_allele:
                    # If we know the ref allele, find entries that contain the alt
                    alt_alleles = alleles_in_genotype - {ref_allele}
                    if not alt_alleles:
                        continue  # This is a homozygous reference entry — skip
                    # Use the het entry (ref + alt) as a proxy for alt allele freq
                    if ref_allele in alleles_in_genotype and len(alleles_in_genotype) == 2:
                        # Heterozygous entry — frequency here IS approximately the alt AF
                        if code not in freq_dict or freq > 0:
                            freq_dict[code] = round(freq, 6)
                else:
                    # No ref allele known — just take the highest non-100% frequency
                    # (the most common non-reference frequency pattern)
                    if freq < 0.99 and freq > 0:
                        if code not in freq_dict:
                            freq_dict[code] = round(freq, 6)

            return freq_dict if freq_dict else None

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return None
        except Exception:
            return None

    return None


# =============================================================================
# STEP 2 FALLBACK: Try the /vep endpoint if population_genotypes gave nothing
# =============================================================================

def fetch_frequencies_vep_fallback(rsid):
    """
    Tries the Ensembl VEP REST endpoint as a fallback.
    VEP annotates variants with gnomAD frequencies from its cache.

    Returns same format: {'afr': 0.015, 'nfe': 0.154, ...} or None.
    """
    url = f"https://rest.ensembl.org/vep/human/id/{rsid}?AF=1&AF_gnomAD=1"

    try:
        response = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code != 200:
            return None

        data = response.json()
        # VEP returns a list of consequence objects. Each may have
        # colocated_variants with frequency data.
        # Structure:
        # [
        #   {
        #     "input": "rs4149056",
        #     "colocated_variants": [
        #       {
        #         "id": "rs4149056",
        #         "frequencies": {
        #           "C": {
        #             "gnomad_afr": 0.015,
        #             "gnomad_eas": 0.143,
        #             "gnomad_nfe": 0.154,
        #             ...
        #           }
        #         }
        #       }
        #     ]
        #   }
        # ]

        freq_dict = {}

        for result in data:
            colocated = result.get('colocated_variants', [])
            for variant in colocated:
                if variant.get('id') != rsid:
                    continue
                frequencies = variant.get('frequencies', {})
                for allele, pop_freqs in frequencies.items():
                    for pop_key, freq in pop_freqs.items():
                        # pop_key looks like "gnomad_afr", "gnomad_nfe", etc.
                        if not pop_key.startswith('gnomad'):
                            continue
                        # Convert "gnomad_afr" → "afr"
                        code = pop_key.replace('gnomadg_', '').replace('gnomad_', '')
                        if '_' in code:
                            continue
                        if freq is not None and freq > 0:
                            freq_dict[code] = round(float(freq), 6)

        return freq_dict if freq_dict else None

    except Exception:
        return None


# =============================================================================
# MAIN FETCH LOOP WITH CHECKPOINTING
# =============================================================================

def fetch_all_frequencies(rsid_map):
    print("\n" + "="*60)
    print("[STEP 2] Fetching population frequencies")
    print("  Source: Ensembl REST API (mirrors gnomAD data)")
    print("="*60)

    freqs_cache = load_checkpoint(CHECKPOINT_FREQS)

    all_rsids = list(rsid_map.keys())
    total = len(all_rsids)
    n_have = sum(1 for r in all_rsids if r in freqs_cache)

    print(f"\n  Total variants:      {total}")
    print(f"  Already fetched:     {n_have}")
    print(f"  Still to fetch:      {total - n_have}")
    print(f"\n  Press Ctrl+C at any time — progress is saved after each variant.")
    print()

    failed = []
    source_counts = {'ensembl_population': 0, 'vep_fallback': 0}

    for i, rsid in enumerate(all_rsids):
        pct = round((i + 1) / total * 100)
        print(f"\r  [{i+1}/{total}] {pct}%  |  {rsid}                    ", end='')
        sys.stdout.flush()

        if rsid in freqs_cache:
            continue  # already have it — skip

        # --- Try primary source: Ensembl population_genotypes ---
        freq_dict = fetch_frequencies_from_ensembl(rsid)
        time.sleep(SLEEP_API)

        if freq_dict:
            source_counts['ensembl_population'] += 1
        else:
            # --- Try fallback: Ensembl VEP ---
            freq_dict = fetch_frequencies_vep_fallback(rsid)
            time.sleep(SLEEP_API)
            if freq_dict:
                source_counts['vep_fallback'] += 1

        if freq_dict:
            freqs_cache[rsid] = freq_dict
            save_checkpoint(CHECKPOINT_FREQS, freqs_cache)
        else:
            failed.append(rsid)
            # Don't save failures to cache — they'll be retried on next run.
            # (Unlike coords where we cache None to avoid re-hitting Ensembl)

    print()
    print(f"\n  Results:")
    print(f"  ✓ Frequencies obtained:       {len(freqs_cache)}")
    print(f"    - via population endpoint:  {source_counts['ensembl_population']}")
    print(f"    - via VEP fallback:         {source_counts['vep_fallback']}")
    print(f"  ✗ Not found in any source:    {len(failed)}")

    if len(failed) > total * 0.5:
        # More than half failed — almost certainly a network/API issue
        print("\n  WARNING: More than 50% of variants failed.")
        print("  This usually means an API issue, not missing data.")
        print("  Things to try:")
        print("    1. Run:  python gnomad_variant_freq.py --diagnose")
        print("    2. Check https://rest.ensembl.org is reachable from your machine")
        print("    3. Wait 10 minutes and try again (rate limiting)")
        print("    4. Check if you're on a restricted network (university VPN, firewall)")

    return freqs_cache


# =============================================================================
# STEP 3: BUILD OUTPUT TABLES
# =============================================================================

def build_output_tables(freqs_cache, rsid_map):
    print("\n" + "="*60)
    print("[STEP 3] Building output tables")
    print("="*60)

    wide_rows = []
    long_rows  = []

    for rsid, freq_dict in freqs_cache.items():
        annotations = rsid_map.get(rsid, [{
            'gene': 'Unknown', 'drug': 'Unknown',
            'evidence': 'Unknown', 'phenotype': 'Unknown'
        }])

        first_ann = annotations[0]

        # WIDE ROW — one per variant, one column per population
        wide_row = {
            'rsid':         rsid,
            'gene':         first_ann['gene'],
            'drug':         first_ann['drug'],
            'evidence':     first_ann['evidence'],
            'phenotype':    first_ann['phenotype'],
        }
        for pop_code in OUTPUT_POPS:
            wide_row[f'freq_{pop_code}'] = freq_dict.get(pop_code)
        wide_rows.append(wide_row)

        # LONG ROWS — one per annotation × population
        for ann in annotations:
            for pop_code in OUTPUT_POPS:
                freq = freq_dict.get(pop_code)
                if freq is None:
                    continue
                # Map short code back to full name
                # Try both key styles (gnomAD:afr and gnomADg:afr)
                pop_name = (
                    POPULATION_MAP.get(f'gnomAD:{pop_code}') or
                    POPULATION_MAP.get(f'gnomADg:{pop_code}') or
                    pop_code.upper()
                )
                long_rows.append({
                    'rsid':            rsid,
                    'gene':            ann['gene'],
                    'drug':            ann['drug'],
                    'evidence':        ann['evidence'],
                    'phenotype':       ann['phenotype'],
                    'population_code': pop_code,
                    'population_name': pop_name,
                    'frequency':       freq,
                })

    wide_df = pd.DataFrame(wide_rows)
    long_df  = pd.DataFrame(long_rows)

    if not wide_df.empty:
        wide_df = wide_df.sort_values(['gene', 'rsid'])
    if not long_df.empty:
        long_df = long_df.sort_values(['gene', 'rsid', 'population_code'])

    print(f"\n  Wide table: {len(wide_df)} rows × {len(wide_df.columns)} columns")
    print(f"  Long table: {len(long_df)} rows")
    return wide_df, long_df


# =============================================================================
# STEP 4: DRUG RESPONSE MAP
# =============================================================================

def build_drug_response_map(long_df):
    print("\n" + "="*60)
    print("[STEP 4] Building drug response map")
    print("="*60)

    if long_df.empty:
        print("  No data.")
        return pd.DataFrame()

    drug_map = long_df.pivot_table(
        index=['rsid', 'gene', 'drug', 'evidence', 'phenotype'],
        columns='population_name',
        values='frequency',
        aggfunc='first'
    ).reset_index()
    drug_map.columns.name = None

    pop_name_values = set(POPULATION_MAP.values())
    pop_cols = [c for c in drug_map.columns if c in pop_name_values]

    if pop_cols:
        drug_map['max_freq']          = drug_map[pop_cols].max(axis=1).round(4)
        drug_map['min_freq']          = drug_map[pop_cols].min(axis=1).round(4)
        drug_map['population_spread'] = (drug_map['max_freq'] - drug_map['min_freq']).round(4)
        drug_map['high_pop_variability'] = drug_map['population_spread'] > 0.10

    drug_map = drug_map.sort_values('population_spread', ascending=False)

    n_total = len(drug_map)
    n_high  = int(drug_map.get('high_pop_variability', pd.Series(dtype=bool)).sum())
    print(f"\n  Variant-drug combinations:   {n_total}")
    print(f"  High population variability: {n_high}")
    return drug_map


# =============================================================================
# STEP 5: SAVE + SUMMARY
# =============================================================================

def save_outputs(wide_df, long_df, drug_map):
    print("\n" + "="*60)
    print("[STEP 5] Saving output files")
    print("="*60)

    if not wide_df.empty:
        wide_df.to_csv(OUTPUT_WIDE, sep='\t', index=False)
        print(f"\n  ✓ {OUTPUT_WIDE}  ({len(wide_df)} variants)")

    if not long_df.empty:
        long_df.to_csv(OUTPUT_LONG, sep='\t', index=False)
        print(f"  ✓ {OUTPUT_LONG}  ({len(long_df)} rows)")

    if not drug_map.empty:
        drug_map.to_csv(OUTPUT_DRUG_MAP, sep='\t', index=False)
        print(f"  ✓ {OUTPUT_DRUG_MAP}  ({len(drug_map)} rows)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if not wide_df.empty:
        print(f"\n  Variants:  {len(wide_df)}")
        print(f"  Genes:     {wide_df['gene'].nunique()}")
        print(f"  Drugs:     {wide_df['drug'].nunique()}")

        print("\n  Evidence levels:")
        for level, count in wide_df.groupby('evidence')['rsid'].nunique().sort_index().items():
            print(f"    {level}: {count} variants")

        print("\n  Top 10 genes:")
        for gene, count in wide_df.groupby('gene')['rsid'].nunique().sort_values(ascending=False).head(10).items():
            print(f"    {gene:<15} {count}")

    if not drug_map.empty and 'population_spread' in drug_map.columns:
        print("\n  Top 10 most population-variable variants:")
        cols = [c for c in ['rsid','gene','drug','evidence','phenotype','population_spread'] if c in drug_map.columns]
        print(drug_map[cols].head(10).to_string(index=False))

    if not long_df.empty and 'rs4149056' in long_df['rsid'].values:
        print("\n  Example — rs4149056 (SLCO1B1 / statins):")
        ex = (long_df[long_df['rsid'] == 'rs4149056']
              [['population_name', 'frequency', 'drug', 'phenotype']]
              .drop_duplicates()
              .sort_values('frequency', ascending=False))
        print(ex.to_string(index=False))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Check for --diagnose flag
    if '--diagnose' in sys.argv:
        run_diagnostic()
        sys.exit(0)

    print("="*60)
    print("  Variant Frequency Calculator (Ensembl/gnomAD)")
    print("  Checkpoint/resume enabled")
    print("="*60)

    try:
        import pandas, requests
        print("\n✓ Libraries OK")
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Run:  pip install pandas requests")
        sys.exit(1)

    rsid_map = load_pharmgkb(PHARMGKB_FILE)
    if not rsid_map:
        sys.exit(1)

    freqs_cache = fetch_all_frequencies(rsid_map)

    if not freqs_cache:
        print("\nNo frequencies obtained.")
        print("Run:  python gnomad_variant_freq.py --diagnose")
        sys.exit(1)

    wide_df, long_df = build_output_tables(freqs_cache, rsid_map)
    drug_map         = build_drug_response_map(long_df)
    save_outputs(wide_df, long_df, drug_map)

    print("\n" + "="*60)
    print("  DONE! Open these files:")
    print(f"    {OUTPUT_WIDE}")
    print(f"    {OUTPUT_LONG}")
    print(f"    {OUTPUT_DRUG_MAP}")
    print("="*60)