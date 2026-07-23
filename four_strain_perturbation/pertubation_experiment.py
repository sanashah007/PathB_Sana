"""
perturbation_experiment.py
==========================
Randomly flip gene presence/absence values in input genomes and run
single_genome_run.py across all flip strategies, sizes, seeds, and
variation parameters to test whether introduced errors produce a
distinct model signature.

Usage:
    python perturbation_experiment.py

Outputs:
    perturbed_inputs/   -- modified .txt genome files
    results/            -- raw output from single_genome_run.py
    perturbation_results.xlsx  -- collated comparison spreadsheet
"""

import os
import sys
import subprocess
import random
import copy
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG — edit these paths to match your cluster
# =============================================================================

STRAINS = {
    "10465": "all_embeded_genomes/GCF_000010465.1_ASM1046v1__254__2007__0000.txt",
    "12045": "all_embeded_genomes/GCF_000012045.1_ASM1204v1__250__2005__0000.txt",
    "13425": "all_embeded_genomes/GCF_000013425.1_ASM1342v1__8__2006__0000.txt",
    "13465": "all_embeded_genomes/GCF_000013465.1_ASM1346v1__8__2006__0000.txt",
}

GENE_NAMES_FILE = "gene_names.txt"

VARIATIONS = [1, 10, 100]

# Flip sizes: number of genes to flip per experiment
FLIP_SIZES = [5, 20, 50]

# Random seeds per (strategy, flip_size) combo
SEEDS = [42, 123, 7]

# Output directories
PERTURBED_DIR = Path("perturbed_inputs")
RESULTS_DIR   = Path("results")
OUTPUT_DIR    = Path("perturbation_output")

# =============================================================================
# GENE CATEGORY LOOKUP  (from reference spreadsheet)
# =============================================================================

def load_gene_categories(gene_names_path=GENE_NAMES_FILE):
    """Return dict: gene_idx (int) -> (gene_name, category)"""

    def categorize(name):
        n = name.lower()
        if any(x in n for x in ['phage','phi ','hk97','capsid','terminase','portal',
                                  'tail','endolysin','integrase','xkd','head-tail',
                                  'lysostaphin','autolysin','tape measure','mu-f']):
            return 'phage'
        elif any(x in n for x in ['ribosom','rna pol','dna pol','dnaa','gyrase',
                                    'topoisomerase','recf','dnab','fts','mur',
                                    '30s','50s','trna','rrna','atp synthase','mnma']):
            return 'core'
        elif any(x in n for x in ['tet(','bcrb','bacitracin','meca','efflux','stk1',
                                    'iscr','msr','macrolide','beta-lacta','blaz',
                                    'blar','blai','erm','type i restriction']):
            return 'resistance'
        elif any(x in n for x in ['iron','isca','iscu','fe-s','fur','iron-regulated']):
            return 'iron_fes'
        else:
            return 'other'

    lookup = {}
    with open(gene_names_path) as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                lookup[idx] = (name, categorize(name))
    return lookup


# =============================================================================
# INPUT FILE I/O
# =============================================================================

def read_genome(path):
    """Read genome txt -> list of [presence, genome_order_idx]"""
    genes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            genes.append([int(parts[0]), int(parts[1])])
    return genes


def write_genome(genes, path):
    """Write genome list back to txt format."""
    with open(path, 'w') as f:
        for presence, goi in genes:
            f.write(f"{presence}\t{goi}\n")


# =============================================================================
# FLIP STRATEGIES
# =============================================================================

def flip_random(genes, n, seed, gene_cats):
    """Flip n randomly chosen genes regardless of category."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(genes)), min(n, len(genes)))
    flipped = copy.deepcopy(genes)
    for i in indices:
        flipped[i][0] = 1 - flipped[i][0]
    return flipped, indices


def flip_targeted(genes, n, seed, gene_cats, target_cat):
    """Flip n genes from a specific category (e.g. 'core', 'phage')."""
    rng = random.Random(seed)
    pool = [i for i, (name, cat) in gene_cats.items()
            if cat == target_cat and i < len(genes)]
    if not pool:
        print(f"  Warning: no genes found for category '{target_cat}', falling back to random")
        return flip_random(genes, n, seed, gene_cats)
    indices = rng.sample(pool, min(n, len(pool)))
    flipped = copy.deepcopy(genes)
    for i in indices:
        flipped[i][0] = 1 - flipped[i][0]
    return flipped, indices


STRATEGIES = {
    "random":        lambda genes, n, seed, cats: flip_random(genes, n, seed, cats),
    "core_targeted": lambda genes, n, seed, cats: flip_targeted(genes, n, seed, cats, 'core'),
    "phage_targeted":lambda genes, n, seed, cats: flip_targeted(genes, n, seed, cats, 'phage'),
}


# =============================================================================
# RUN PIPELINE
# =============================================================================

def run_single_genome(input_path, variation, results_dir):
    """
    Run single_genome_run.py with output written to results_dir.
    Returns tsv_path on success, None on error.
    """
    stem = Path(input_path).stem
    out_tsv = results_dir / f"{stem}_var{variation}.tsv"

    if out_tsv.exists():
        return out_tsv

    cmd = [sys.executable, "single_genome_run.py",
           "--input", str(input_path),
           "--predict-variation", str(variation),
           "--out", str(out_tsv)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR running {cmd}")
        print(result.stderr)
        return None

    return out_tsv


def parse_output(tsv_path):
    """Read single_genome_run.py output TSV -> DataFrame with gene_idx and logit_prob."""
    if tsv_path is None or not Path(tsv_path).exists():
        return None
    df = pd.read_csv(tsv_path, sep='\t')
    return df[['gene_idx', 'logit_prob']]


# =============================================================================
# PER-GENE DOR ANALYSIS
# =============================================================================

def _compute_dor(df, output_dir):
    print("Computing per-gene DOR analysis ...")
    df = df.copy()
    df['signed_score'] = df['logit_prob'] * ((df['orig_presence'] * 2) - 1)

    cutoff_rows = []
    for gene_idx, grp in df[df['was_flipped']].groupby('gene_idx'):
        cutoff = np.percentile(grp['signed_score'], 99)
        cutoff_rows.append({
            'gene_idx':    gene_idx,
            'gene_name':   grp['gene_name'].iloc[0],
            'category':    grp['category'].iloc[0],
            'n_flipped':   len(grp),
            'cutoff_99th': cutoff,
        })
    per_gene_cutoffs = pd.DataFrame(cutoff_rows)

    cutoff_map = dict(zip(per_gene_cutoffs['gene_idx'], per_gene_cutoffs['cutoff_99th']))
    dor_rows = []
    for gene_idx, grp in df[df['gene_idx'].isin(cutoff_map)].groupby('gene_idx'):
        cutoff   = cutoff_map[gene_idx]
        detected = grp['signed_score'] < cutoff
        flipped  = grp['was_flipped']
        tp = int(( detected &  flipped).sum())
        fp = int(( detected & ~flipped).sum())
        fn = int((~detected &  flipped).sum())
        tn = int((~detected & ~flipped).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fpr         = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        denom       = fp * fn
        dor         = (tp * tn) / denom if denom > 0 else np.nan
        log10_dor   = np.log10(dor) if (not np.isnan(dor) and dor > 0) else np.nan

        dor_rows.append({
            'gene_idx':            gene_idx,
            'gene_name':           grp['gene_name'].iloc[0],
            'category':            grp['category'].iloc[0],
            'cutoff_99th':         cutoff,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'sensitivity':         sensitivity,
            'specificity':         specificity,
            'false_positive_rate': fpr,
            'dor':                 dor,
            'log10_dor':           log10_dor,
        })

    per_gene_dor = (pd.DataFrame(dor_rows)
                    .sort_values('log10_dor', ascending=False)
                    .reset_index(drop=True))

    per_gene_cutoffs.to_csv(output_dir / "per_gene_cutoffs.csv", index=False)
    per_gene_dor.to_csv(output_dir / "per_gene_dor.csv", index=False)
    print(f"  {output_dir}/per_gene_cutoffs.csv — {len(per_gene_cutoffs):,} genes")
    print(f"  {output_dir}/per_gene_dor.csv     — {len(per_gene_dor):,} genes")


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_results_path = OUTPUT_DIR / "all_results.csv"
    if all_results_path.exists():
        print(f"Fast-path: loading {all_results_path} — skipping model runs.")
        df = pd.read_csv(all_results_path)
        _compute_dor(df, OUTPUT_DIR)
        return

    PERTURBED_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading gene categories from gene_names.txt ...")
    gene_cats = load_gene_categories(GENE_NAMES_FILE)
    print(f"  Loaded {len(gene_cats)} gene categories")

    all_records = []

    for strain_id, genome_path in STRAINS.items():
        if not os.path.exists(genome_path):
            print(f"[SKIP] {genome_path} not found — check STRAINS config")
            continue

        print(f"\n{'='*60}")
        print(f"Strain {strain_id}: {genome_path}")

        baseline_genes = read_genome(genome_path)
        n_genes = len(baseline_genes)
        n_present = sum(g[0] for g in baseline_genes)
        print(f"  {n_genes} genes, {n_present} present ({n_present/n_genes:.1%})")

        # --- Baseline runs (needed for delta computation) ---
        baseline_logits = {}   # variation -> {gene_idx: logit_prob}
        for variation in VARIATIONS:
            print(f"  Baseline variation={variation} ...")
            out_file = run_single_genome(genome_path, variation, RESULTS_DIR)
            if out_file is None:
                print(f"    Baseline failed for var={variation}")
                continue
            df_base = parse_output(out_file)
            baseline_logits[variation] = dict(zip(df_base['gene_idx'], df_base['logit_prob']))

        # --- Perturbation runs ---
        for strategy_name, strategy_fn in STRATEGIES.items():
            for n_flips in FLIP_SIZES:
                for seed in SEEDS:
                    exp_id = f"{strain_id}_{strategy_name}_n{n_flips}_s{seed}"
                    print(f"  {exp_id} ...")

                    flipped_genes, flipped_indices = strategy_fn(
                        baseline_genes, n_flips, seed, gene_cats)
                    flipped_set = set(flipped_indices)

                    # Write perturbed input (skip if already exists)
                    perturbed_path = PERTURBED_DIR / f"{exp_id}.txt"
                    if not perturbed_path.exists():
                        write_genome(flipped_genes, perturbed_path)

                    for variation in VARIATIONS:
                        out_file = run_single_genome(
                            perturbed_path, variation, RESULTS_DIR)
                        if out_file is None:
                            print(f"    No output for var={variation}")
                            continue
                        df_out = parse_output(out_file)

                        base_var = baseline_logits.get(variation, {})
                        for _, row in df_out.iterrows():
                            idx = int(row['gene_idx'])
                            gene_name, cat = gene_cats.get(idx, ('unknown', 'unknown'))
                            orig = baseline_genes[idx][0] if idx < n_genes else None
                            new  = flipped_genes[idx][0]  if idx < n_genes else None
                            base_logit = base_var.get(idx)
                            base_suspect = (
                                ((orig == 0) and (base_logit > 0)) or
                                ((orig == 1) and (base_logit < 0))
                            ) if base_logit is not None else None
                            all_records.append({
                                'strain':           strain_id,
                                'strategy':         strategy_name,
                                'flip_size':        n_flips,
                                'seed':             seed,
                                'variation':        variation,
                                'gene_idx':         idx,
                                'gene_name':        gene_name,
                                'category':         cat,
                                'was_flipped':      idx in flipped_set,
                                'orig_presence':    orig,
                                'new_presence':     new,
                                'logit_prob':       row['logit_prob'],
                                'baseline_logit':   base_logit,
                                'baseline_suspect': base_suspect,
                            })

    # =========================================================================
    # BUILD OUTPUT SPREADSHEET
    # =========================================================================
    if not all_records:
        print("\nNo results collected — check that single_genome_run.py runs correctly.")
        return

    df = pd.DataFrame(all_records)

    # Compute z-score within (strain, variation, orig_presence) group
    def add_z(group):
        mu, sd = group['logit_prob'].mean(), group['logit_prob'].std()
        group = group.copy()
        group['z_score'] = (group['logit_prob'] - mu) / sd if sd > 0 else 0.0
        return group

    df = df.groupby(['strain','variation','orig_presence'], group_keys=False).apply(add_z)

    # Flag: is this a suspect mismatch (wrong direction)?
    df['suspect'] = (
        ((df['orig_presence'] == 0) & (df['logit_prob'] > 0)) |
        ((df['orig_presence'] == 1) & (df['logit_prob'] < 0))
    )

    # ---- Summary pivot: for flipped genes, compare logit across var 1/10/100 ----
    flipped_df = df[df['was_flipped']].copy()

    pivot = flipped_df.pivot_table(
        index=['strain','strategy','flip_size','seed','gene_idx','gene_name',
               'category','orig_presence','new_presence'],
        columns='variation',
        values=['logit_prob','z_score','suspect','baseline_logit','baseline_suspect'],
        aggfunc='first'
    )
    pivot.columns = [f'{col[0]}_var{col[1]}' for col in pivot.columns]
    pivot = pivot.reset_index()

    # Consistency flag: is the suspect signal the same across all 3 variations?
    for v in VARIATIONS:
        if f'suspect_var{v}' in pivot.columns:
            pivot[f'suspect_var{v}'] = pivot[f'suspect_var{v}'].fillna(False)
        if f'baseline_suspect_var{v}' in pivot.columns:
            pivot[f'baseline_suspect_var{v}'] = pivot[f'baseline_suspect_var{v}'].fillna(False)

    if all(f'suspect_var{v}' in pivot.columns for v in VARIATIONS):
        pivot['consistent_across_variations'] = (
            pivot[['suspect_var1','suspect_var10','suspect_var100']].nunique(axis=1) == 1
        )
        pivot['only_suspect_at_100'] = (
            ~pivot['suspect_var1'] & ~pivot['suspect_var10'] & pivot['suspect_var100']
        )
        pivot['suspect_at_all'] = (
            pivot['suspect_var1'] & pivot['suspect_var10'] & pivot['suspect_var100']
        )

    # Add delta and newly_suspect per variation
    for v in VARIATIONS:
        lp_col = f'logit_prob_var{v}'
        bl_col = f'baseline_logit_var{v}'
        bs_col = f'baseline_suspect_var{v}'
        sp_col = f'suspect_var{v}'
        if lp_col in pivot.columns and bl_col in pivot.columns:
            pivot[f'delta_var{v}'] = pivot[lp_col] - pivot[bl_col]
        if sp_col in pivot.columns and bs_col in pivot.columns:
            pivot[f'newly_suspect_var{v}'] = pivot[sp_col] & ~pivot[bs_col]

    # delta_summary: group by (strategy, category, flip_size, variation)
    delta_records = []
    for v in VARIATIONS:
        d_col  = f'delta_var{v}'
        ns_col = f'newly_suspect_var{v}'
        if d_col not in pivot.columns:
            continue
        grp = pivot.groupby(['strategy', 'category', 'flip_size']).agg(
            n_genes          = ('gene_idx',  'count'),
            mean_abs_delta   = (d_col,       lambda x: x.abs().mean()),
            pct_newly_suspect= (ns_col,      'mean') if ns_col in pivot.columns else ('gene_idx', 'count'),
        ).reset_index()
        grp.insert(0, 'variation', v)
        delta_records.append(grp)
    delta_summary = pd.concat(delta_records, ignore_index=True) if delta_records else pd.DataFrame()

    print(f"\nWriting CSVs to {OUTPUT_DIR}/ ...")

    df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    pivot.to_csv(OUTPUT_DIR / "flipped_gene_summary.csv", index=False)

    if not pivot.empty and 'consistent_across_variations' in pivot.columns:
        summary = pivot.groupby(['strategy','flip_size','category']).agg(
            n_genes=('gene_idx','count'),
            pct_consistent=('consistent_across_variations','mean'),
            pct_suspect_all=('suspect_at_all','mean') if 'suspect_at_all' in pivot.columns else ('gene_idx','count'),
            pct_only_at_100=('only_suspect_at_100','mean') if 'only_suspect_at_100' in pivot.columns else ('gene_idx','count'),
        ).reset_index()
        summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)

    if not delta_summary.empty:
        delta_summary.to_csv(OUTPUT_DIR / "delta_summary.csv", index=False)

    print(f"Done.")
    print(f"  {OUTPUT_DIR}/all_results.csv          — {len(df):,} rows")
    print(f"  {OUTPUT_DIR}/flipped_gene_summary.csv — {len(pivot):,} rows")
    print(f"  {OUTPUT_DIR}/delta_summary.csv         — {len(delta_summary):,} rows")

    _compute_dor(df, OUTPUT_DIR)


if __name__ == '__main__':
    main()