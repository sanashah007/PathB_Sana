#!/usr/bin/env python3
"""
metagenome_perturbation.py
==========================
Apply the same random-flip perturbation experiment used in population_perturbation.py
to metagenome feature vectors, using pre-computed train-set cutoffs from the reference
population so results are on the same scale and directly comparable.

Differences from population_perturbation.py:
  - No train pass: loads per_gene_cutoffs.tsv from the reference experiment
  - No train/val/test split: all metagenomes are the evaluation population
  - Same flip_size, seeds, and variations as the reference run by default

Output files (same format as population_perturbation.py):
  per_gene_dor.tsv            — per-gene DOR across all metagenomes (pooled variations)
  per_gene_dor_by_variation.tsv — per-gene DOR broken out by variation value
  summary.txt                 — run metadata and high-level stats

Must be run from PATHB_sample root where run_model.py and twolayer.pth reside.

Usage:
    python metagenome_tests/metagenome_perturbation.py \\
        --vectors-dir metagenome_tests/metagenome_vectors/ \\
        --cutoffs     population_perturbation/population_perturbation_results/per_gene_cutoffs.tsv \\
        --out-dir     metagenome_tests/metagenome_perturbation_results/
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_model as _rm


# =============================================================================
# GENE CATEGORY LOOKUP  (copied verbatim from population_perturbation.py)
# =============================================================================

def _categorize(name):
    n = name.lower()
    if any(x in n for x in ['phage', 'phi ', 'hk97', 'capsid', 'terminase', 'portal',
                              'tail', 'endolysin', 'integrase', 'xkd', 'head-tail',
                              'lysostaphin', 'autolysin', 'tape measure', 'mu-f']):
        return 'phage'
    elif any(x in n for x in ['ribosom', 'rna pol', 'dna pol', 'dnaa', 'gyrase',
                                'topoisomerase', 'recf', 'dnab', 'fts', 'mur',
                                '30s', '50s', 'trna', 'rrna', 'atp synthase', 'mnma']):
        return 'core'
    elif any(x in n for x in ['tet(', 'bcrb', 'bacitracin', 'meca', 'efflux', 'stk1',
                                'iscr', 'msr', 'macrolide', 'beta-lacta', 'blaz',
                                'blar', 'blai', 'erm', 'type i restriction']):
        return 'resistance'
    elif any(x in n for x in ['iron', 'isca', 'iscu', 'fe-s', 'fur', 'iron-regulated']):
        return 'iron_fes'
    else:
        return 'other'


def load_gene_categories(gene_names_path):
    lookup = {}
    with open(gene_names_path) as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                lookup[idx] = (name, _categorize(name))
    return lookup


# =============================================================================
# MODEL INTERFACE
# =============================================================================

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def run_model_batched(X, num_not_shared):
    result = _rm.run_model(X, num_not_shared=num_not_shared)
    if hasattr(result, 'detach'):
        result = result.detach().cpu().numpy()
    return np.asarray(result, dtype=np.float32)


# =============================================================================
# GENOME I/O
# =============================================================================

def load_vectors(paths):
    arrays = [np.loadtxt(p, usecols=[0], dtype=np.uint8) for p in paths]
    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent vector lengths: {sorted(lengths)}")
    return np.stack(arrays, axis=0)


# =============================================================================
# BATCH BUILDER  (identical to population_perturbation.py)
# =============================================================================

def _build_batch(X_clean, flip_size, seeds):
    N, G = X_clean.shape
    X_parts, mask_parts = [], []
    for seed in seeds:
        rng       = np.random.default_rng(seed)
        X_corrupt = X_clean.copy()
        error_mask = np.zeros((N, G), dtype=bool)
        for i in range(N):
            idx = rng.choice(G, size=min(flip_size, G), replace=False)
            X_corrupt[i, idx]  = 1 - X_corrupt[i, idx]
            error_mask[i, idx] = True
        X_parts.append(X_corrupt)
        mask_parts.append(error_mask)
    return np.vstack(X_parts), np.vstack(mask_parts)


# =============================================================================
# DOR HELPERS  (identical to population_perturbation.py)
# =============================================================================

def _dor_from_counts(tp, fp, fn, tn, nf, nnf, cutoff):
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    sensitivity = tp / (tp + fn)    if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp)    if (tn + fp) > 0 else float('nan')
    fpr         = fp / (fp + tn)    if (fp + tn) > 0 else float('nan')
    denom       = fp * fn
    dor         = (tp * tn) / denom if denom > 0       else float('nan')
    log10_dor   = float(np.log10(dor)) if (not np.isnan(dor) and dor > 0) else float('nan')
    return dict(
        n_flipped=int(nf), n_not_flipped=int(nnf),
        cutoff_99th=cutoff,
        tp=tp, fp=fp, fn=fn, tn=tn,
        sensitivity=sensitivity, specificity=specificity,
        false_positive_rate=fpr, dor=dor, log10_dor=log10_dor,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Metagenome perturbation experiment — comparable to population_perturbation.py.'
    )
    p.add_argument('--vectors-dir', default='metagenome_tests/metagenome_vectors/',
                   help='Directory of metagenome .txt vectors')
    p.add_argument('--cutoffs',
                   default='population_perturbation/population_perturbation_results/per_gene_cutoffs.tsv',
                   help='per_gene_cutoffs.tsv from the reference population_perturbation run')
    p.add_argument('--genes',       default='gene_names.txt')
    p.add_argument('--flip-size',   type=int, default=20,
                   help='Genes to flip per genome per seed (default 20, same as reference)')
    p.add_argument('--seeds',       default='42,123,7',
                   help='RNG seeds (default same as reference run)')
    p.add_argument('--variations',  default='1,10,100',
                   help='num_not_shared values (default same as reference run)')
    p.add_argument('--min-val-errors', type=int, default=1,
                   help='Min flipped instances required to report DOR for a gene')
    p.add_argument('--out-dir',
                   default='metagenome_tests/metagenome_perturbation_results/',
                   help='Output directory')
    args = p.parse_args()

    seeds      = [int(s) for s in args.seeds.split(',')]
    variations = [int(v) for v in args.variations.split(',')]
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load gene categories
    # -------------------------------------------------------------------------
    print("Loading gene categories ...")
    gene_cats = load_gene_categories(args.genes)
    G = len(gene_cats)
    print(f"  {G} genes")

    # -------------------------------------------------------------------------
    # Load reference cutoffs
    # -------------------------------------------------------------------------
    print(f"Loading reference cutoffs from {args.cutoffs} ...")
    cutoffs_df = pd.read_csv(args.cutoffs, sep='\t')
    cutoff_vec = np.full(G, np.nan, dtype=np.float64)
    for _, row in cutoffs_df.iterrows():
        idx = int(row['gene_idx'])
        if 0 <= idx < G:
            cutoff_vec[idx] = float(row['cutoff_99th'])
    n_valid = int(np.sum(~np.isnan(cutoff_vec)))
    print(f"  {n_valid} / {G} genes have valid cutoffs")

    # -------------------------------------------------------------------------
    # Load metagenome vectors
    # -------------------------------------------------------------------------
    vectors_dir  = Path(args.vectors_dir)
    vector_files = sorted(vectors_dir.glob('*.txt'))
    if not vector_files:
        sys.exit(f'No .txt files found in {vectors_dir}')
    print(f"\nLoading {len(vector_files)} metagenome vectors ...")
    X = load_vectors(vector_files)
    print(f"  shape: {X.shape}")

    # -------------------------------------------------------------------------
    # Build corrupted batches  (same procedure as population_perturbation.py)
    # -------------------------------------------------------------------------
    print("\nBuilding corrupted batches ...")
    X_all, mask_all = _build_batch(X, args.flip_size, seeds)
    print(f"  corrupted shape: {X_all.shape}  "
          f"(n_genomes={len(vector_files)}, n_seeds={len(seeds)})")
    total_flips = int(mask_all.sum())
    avg_flips_per_gene = total_flips / G
    print(f"  total flip events: {total_flips}  "
          f"avg per gene: {avg_flips_per_gene:.1f}")

    # -------------------------------------------------------------------------
    # Accumulate TP/FP/FN/TN using reference cutoffs
    # -------------------------------------------------------------------------
    print("\nRunning model and accumulating counts ...")

    zeros    = lambda: np.zeros(G, dtype=np.int64)
    tp_pool  = zeros(); fp_pool  = zeros()
    fn_pool  = zeros(); tn_pool  = zeros()
    nf_pool  = zeros(); nnf_pool = zeros()

    tp_var  = {v: zeros() for v in variations}
    fp_var  = {v: zeros() for v in variations}
    fn_var  = {v: zeros() for v in variations}
    tn_var  = {v: zeros() for v in variations}
    nf_var  = {v: zeros() for v in variations}
    nnf_var = {v: zeros() for v in variations}

    sign_all  = (X_all.astype(np.float32) * 2.0) - 1.0
    valid_idx = ~np.isnan(cutoff_vec)

    for v in variations:
        print(f"  variation={v} ...")
        probs  = run_model_batched(X_all, num_not_shared=v)
        scores = _logit(probs) * sign_all
        del probs

        # Pooled counts
        s   = scores[:, valid_idx]
        cv  = cutoff_vec[valid_idx]
        m   = mask_all[:, valid_idx]
        det = s < cv[None, :]
        tp_pool[valid_idx]  += ( m  &  det).sum(0)
        fn_pool[valid_idx]  += ( m  & ~det).sum(0)
        fp_pool[valid_idx]  += (~m  &  det).sum(0)
        tn_pool[valid_idx]  += (~m  & ~det).sum(0)
        nf_pool[valid_idx]  +=   m.sum(0)
        nnf_pool[valid_idx] += (~m).sum(0)

        # Per-variation counts (same cutoffs — reference only provides pooled)
        tp_var[v][valid_idx]  += ( m  &  det).sum(0)
        fn_var[v][valid_idx]  += ( m  & ~det).sum(0)
        fp_var[v][valid_idx]  += (~m  &  det).sum(0)
        tn_var[v][valid_idx]  += (~m  & ~det).sum(0)
        nf_var[v][valid_idx]  +=   m.sum(0)
        nnf_var[v][valid_idx] += (~m).sum(0)

        del s, cv, m, det, scores

    del X_all, mask_all, sign_all
    print("  Done.")

    # -------------------------------------------------------------------------
    # Build DOR rows
    # -------------------------------------------------------------------------
    print("\nComputing per-gene DOR ...")

    dor_rows = []
    for gene_idx in range(G):
        cutoff = cutoff_vec[gene_idx]
        if np.isnan(cutoff):
            continue
        nf  = int(nf_pool[gene_idx])
        nnf = int(nnf_pool[gene_idx])
        if nf < args.min_val_errors:
            continue
        gene_name, category = gene_cats.get(gene_idx, ('unknown', 'unknown'))
        stats = _dor_from_counts(
            tp_pool[gene_idx], fp_pool[gene_idx],
            fn_pool[gene_idx], tn_pool[gene_idx],
            nf, nnf, cutoff,
        )
        dor_rows.append({'gene_idx': gene_idx, 'gene_name': gene_name,
                         'category': category, **stats})

    per_gene_dor = (
        pd.DataFrame(dor_rows)
        .sort_values('log10_dor', ascending=False, na_position='last')
        .reset_index(drop=True)
    )

    var_dor_rows = []
    for v in variations:
        for gene_idx in range(G):
            cutoff = cutoff_vec[gene_idx]
            if np.isnan(cutoff):
                continue
            nf  = int(nf_var[v][gene_idx])
            nnf = int(nnf_var[v][gene_idx])
            if nf < args.min_val_errors:
                continue
            gene_name, category = gene_cats.get(gene_idx, ('unknown', 'unknown'))
            stats = _dor_from_counts(
                tp_var[v][gene_idx], fp_var[v][gene_idx],
                fn_var[v][gene_idx], tn_var[v][gene_idx],
                nf, nnf, cutoff,
            )
            var_dor_rows.append({'variation': v, 'gene_idx': gene_idx,
                                  'gene_name': gene_name, 'category': category,
                                  **stats})

    per_gene_dor_by_var = (
        pd.DataFrame(var_dor_rows)
        .sort_values(['variation', 'log10_dor'], ascending=[True, False], na_position='last')
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    per_gene_dor.to_csv(out_dir / 'per_gene_dor.tsv', sep='\t', index=False)
    per_gene_dor_by_var.to_csv(out_dir / 'per_gene_dor_by_variation.tsv', sep='\t', index=False)

    with open(out_dir / 'summary.txt', 'w') as fh:
        fh.write("Metagenome Perturbation Summary\n")
        fh.write("================================\n")
        fh.write(f"  Vectors dir:       {args.vectors_dir}\n")
        fh.write(f"  Reference cutoffs: {args.cutoffs}\n")
        fh.write(f"  N metagenomes:     {len(vector_files)}\n")
        fh.write(f"  Genes:             {G}\n")
        fh.write(f"  Valid cutoffs:     {n_valid}\n")
        fh.write(f"  Flip size:         {args.flip_size}\n")
        fh.write(f"  Seeds:             {seeds}\n")
        fh.write(f"  Variations:        {variations}\n")
        fh.write(f"  Total flip events: {total_flips}\n")
        fh.write(f"  Avg flips/gene:    {avg_flips_per_gene:.2f}\n")
        fh.write(f"\n  per_gene_dor rows:           {len(per_gene_dor)}\n")
        fh.write(f"  per_gene_dor_by_var rows:    {len(per_gene_dor_by_var)}\n")
        if not per_gene_dor.empty and 'log10_dor' in per_gene_dor.columns:
            top5 = per_gene_dor.dropna(subset=['log10_dor']).head(5)
            fh.write("\n  Top 5 genes by log10_dor (pooled):\n")
            for _, row in top5.iterrows():
                fh.write(f"    [{row['gene_idx']}] {row['gene_name'][:60]}  "
                         f"log10_dor={row['log10_dor']:.3f}\n")

    print(f"\nDone.")
    print(f"  per_gene_dor.tsv          — {len(per_gene_dor)} genes")
    print(f"  per_gene_dor_by_variation.tsv — {len(per_gene_dor_by_var)} rows")
    print(f"  summary.txt")


if __name__ == '__main__':
    main()
