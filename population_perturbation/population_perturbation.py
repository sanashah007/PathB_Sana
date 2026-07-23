#!/usr/bin/env python3
"""
population_perturbation.py
==========================
Run a random-flip perturbation experiment across the full genome population
and compute per-gene diagnostic odds ratios (DOR) as the primary output.

Train / val / test design
--------------------------
  Train (year < 2020)  — used ONLY to compute per-gene score cutoffs
  Val   (year == 2020) — used ONLY to evaluate TP/FP/FN/TN against those cutoffs
  Test  (year > 2020)  — never touched

This avoids the circularity of fitting and evaluating on the same data.

Model interface
---------------
Calls run_model directly (no subprocess overhead). One batched call per
variation level covers all corrupted genomes at once.

Usage:
    python population_perturbation.py [options]
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_model as _rm  # direct import — avoids per-genome subprocess calls


# =============================================================================
# GENE CATEGORY LOOKUP  (categorize copied exactly from perturbation_experiment.py)
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
    """Return dict: gene_idx (int) -> (gene_name, category)"""
    lookup = {}
    with open(gene_names_path) as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                lookup[idx] = (name, _categorize(name))
    return lookup


# =============================================================================
# GENOME I/O
# =============================================================================

def load_vectors(paths):
    """Load genome .txt files -> (N, G) uint8 array (column 0 = presence)."""
    arrays = [np.loadtxt(p, usecols=[0], dtype=np.uint8) for p in paths]
    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise RuntimeError(f"Genome vectors have inconsistent lengths: {sorted(lengths)}")
    return np.stack(arrays, axis=0)


# =============================================================================
# MODEL INTERFACE  (direct batched call — no subprocess)
# =============================================================================

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def run_model_batched(X, num_not_shared):
    """Call run_model directly on a (N, G) batch. Returns numpy (N, G) probs."""
    result = _rm.run_model(X, num_not_shared=num_not_shared)
    if hasattr(result, 'detach'):
        result = result.detach().cpu().numpy()
    return np.asarray(result, dtype=np.float32)


# =============================================================================
# FILENAME PARSING AND TRAIN/VAL/TEST SPLITTING
# =============================================================================

def parse_genome_filename(path):
    """
    Parse <name>__<ST>__<uploadYear>__<collectionYear>.txt
    Returns (upload_year, collection_year) as ints, or (None, None).
    """
    stem = Path(path).stem
    parts = stem.split('__')
    if len(parts) < 4:
        return None, None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None, None


def assign_split(upload_year, collection_year):
    """
    Use the earlier of the two valid years (ignoring zeros) to assign split.
    Defaults to 'train' if no valid year found.
    """
    valid = [y for y in [upload_year, collection_year] if y and y > 0]
    if not valid:
        return 'train'
    year = min(valid)
    if year < 2020:
        return 'train'
    elif year == 2020:
        return 'val'
    else:
        return 'test'


# =============================================================================
# SCORE ACCUMULATION HELPER
# =============================================================================

def _accumulate_errors(X_batch, mask_batch, variations, err_acc, err_acc_var):
    """Accumulate error (flipped-gene) scores for cutoff computation."""
    G    = X_batch.shape[1]
    sign = (X_batch.astype(np.float32) * 2.0) - 1.0

    for v in variations:
        print(f"    variation={v} ...")
        probs  = run_model_batched(X_batch, num_not_shared=v)
        scores = _logit(probs) * sign

        for g in range(G):
            err = scores[mask_batch[:, g], g]
            if err.size > 0:
                err_acc[g].extend(err.tolist())
                err_acc_var[v][g].extend(err.tolist())


# =============================================================================
# BATCH BUILDER
# =============================================================================

def _build_batch(X_clean, flip_size, seeds):
    """
    Apply random flips for each seed and stack into one big batch.
    Returns (X_all, mask_all) each of shape (N * n_seeds, G).
    """
    N, G = X_clean.shape
    X_parts, mask_parts = [], []
    for seed in seeds:
        rng        = np.random.default_rng(seed)
        X_corrupt  = X_clean.copy()
        error_mask = np.zeros((N, G), dtype=bool)
        for i in range(N):
            idx = rng.choice(G, size=min(flip_size, G), replace=False)
            X_corrupt[i, idx]  = 1 - X_corrupt[i, idx]
            error_mask[i, idx] = True
        X_parts.append(X_corrupt)
        mask_parts.append(error_mask)
    return np.vstack(X_parts), np.vstack(mask_parts)


# =============================================================================
# DOR HELPERS
# =============================================================================

def _compute_cutoff(err_list):
    """99th-percentile of a gene's error scores (from train)."""
    return float(np.percentile(err_list, 99))


def _dor_from_counts(tp, fp, fn, tn, nf, nnf, cutoff):
    """Compute DOR stats from pre-counted TP/FP/FN/TN integers."""
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    sensitivity = tp / (tp + fn)    if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp)    if (tn + fp) > 0 else float('nan')
    fpr         = fp / (fp + tn)    if (fp + tn) > 0 else float('nan')
    denom       = fp * fn
    dor         = (tp * tn) / denom if denom > 0       else float('nan')
    log10_dor   = float(np.log10(dor)) if (not np.isnan(dor) and dor > 0) else float('nan')
    return dict(
        n_val_flipped=int(nf), n_val_not_flipped=int(nnf),
        cutoff_99th=cutoff,
        tp=tp, fp=fp, fn=fn, tn=tn,
        sensitivity=sensitivity, specificity=specificity,
        false_positive_rate=fpr, dor=dor, log10_dor=log10_dor,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Population-scale perturbation experiment with per-gene DOR analysis."
    )
    parser.add_argument('--data',                default='all_embeded_genomes/',
                        help='Directory of genome .txt files')
    parser.add_argument('--genes',               default='gene_names.txt',
                        help='gene_names.txt (one name per row)')
    parser.add_argument('--k',         type=int, default=100000,
                        help='Max train genomes to sample for cutoff computation')
    parser.add_argument('--k-val',     type=int, default=100000,
                        help='Max val genomes to sample for DOR evaluation')
    parser.add_argument('--flip-size', type=int, default=20,
                        help='Number of genes to flip per genome per seed')
    parser.add_argument('--seeds',               default='42,123,7',
                        help='Comma-separated RNG seeds for flipping')
    parser.add_argument('--variations',          default='1,10,100',
                        help='Comma-separated predict-variation (num_not_shared) values')
    parser.add_argument('--min-errors-per-gene', type=int, default=5,
                        help='Min train-flipped instances a gene needs for its cutoff to be computed')
    parser.add_argument('--min-val-errors',      type=int, default=1,
                        help='Min val-flipped instances required to report DOR for a gene')
    parser.add_argument('--output-dir',          default='population_perturbation_results/',
                        help='Directory for all output files')
    parser.add_argument('--seed',      type=int, default=42,
                        help='RNG seed for genome sampling')
    args = parser.parse_args()

    seeds      = [int(s) for s in args.seeds.split(',')]
    variations = [int(v) for v in args.variations.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Load gene categories
    # -------------------------------------------------------------------------
    print("Loading gene categories ...")
    gene_cats = load_gene_categories(args.genes)
    G = len(gene_cats)
    print(f"  {G} genes loaded")

    # -------------------------------------------------------------------------
    # Discover genomes and assign splits — keep train and val separate
    # -------------------------------------------------------------------------
    data_dir = Path(args.data)
    all_files = sorted(data_dir.glob('*.txt'))
    print(f"\nDiscovered {len(all_files)} genome files in {data_dir}")

    split_rows  = []
    train_paths = []
    val_paths   = []

    for p in all_files:
        uy, cy = parse_genome_filename(p)
        split = assign_split(uy, cy)
        split_rows.append({
            'filename':        p.name,
            'split':           split,
            'upload_year':     uy if uy else '',
            'collection_year': cy if cy else '',
        })
        if split == 'train':
            train_paths.append(p)
        elif split == 'val':
            val_paths.append(p)
        # test paths are intentionally ignored

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(output_dir / 'train_val_test_split.tsv', sep='\t', index=False)

    n_train = len(train_paths)
    n_val   = len(val_paths)
    n_test  = int((split_df['split'] == 'test').sum())
    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test}  (test never sampled)")

    # -------------------------------------------------------------------------
    # Sample train and val separately
    # -------------------------------------------------------------------------
    rng_sample  = random.Random(args.seed)
    k_train     = min(args.k,     len(train_paths))
    k_val       = min(args.k_val, len(val_paths))
    train_sampled = rng_sample.sample(train_paths, k_train)
    val_sampled   = rng_sample.sample(val_paths,   k_val)
    print(f"\n  Train sampled: {k_train} / {n_train}")
    print(f"  Val   sampled: {k_val}   / {n_val}")

    # -------------------------------------------------------------------------
    # Load genome vectors
    # -------------------------------------------------------------------------
    print(f"\nLoading train vectors ({k_train} genomes) ...")
    X_train = load_vectors(train_sampled)
    print(f"  shape: {X_train.shape}")

    print(f"Loading val vectors ({k_val} genomes) ...")
    X_val = load_vectors(val_sampled)
    print(f"  shape: {X_val.shape}")

    # -------------------------------------------------------------------------
    # Build corrupted batches (all seeds stacked)
    # -------------------------------------------------------------------------
    print("\nBuilding corrupted batches ...")
    X_train_all, mask_train_all = _build_batch(X_train, args.flip_size, seeds)
    X_val_all,   mask_val_all   = _build_batch(X_val,   args.flip_size, seeds)
    avg_flips = int(mask_train_all.sum()) / (k_train * len(seeds))

    # -------------------------------------------------------------------------
    # TRAIN PASS — error scores only (cutoffs)
    # -------------------------------------------------------------------------
    print("\nTrain pass (computing cutoffs) ...")
    err_train     = defaultdict(list)
    err_train_var = {v: defaultdict(list) for v in variations}

    _accumulate_errors(X_train_all, mask_train_all, variations,
                       err_train, err_train_var)
    del X_train_all, mask_train_all

    # Build cutoff vectors directly from train error scores
    cutoff_vec_pool = np.full(G, np.nan, dtype=np.float64)
    for gene_idx, train_err in err_train.items():
        if len(train_err) >= args.min_errors_per_gene:
            cutoff_vec_pool[gene_idx] = _compute_cutoff(train_err)

    cutoff_vecs_var = {}
    for v in variations:
        cv = np.full(G, np.nan, dtype=np.float64)
        for gene_idx, train_err in err_train_var[v].items():
            if len(train_err) >= args.min_errors_per_gene:
                cv[gene_idx] = _compute_cutoff(train_err)
        cutoff_vecs_var[v] = cv

    # Store train flipped counts then free the score lists
    n_train_flipped_pool = np.array([len(err_train[g]) for g in range(G)], dtype=np.int64)
    n_train_flipped_var  = {v: np.array([len(err_train_var[v][g]) for g in range(G)], dtype=np.int64)
                             for v in variations}
    del err_train, err_train_var

    n_valid = int(np.sum(~np.isnan(cutoff_vec_pool)))
    print(f"  {n_valid} / {G} genes have valid cutoffs (pooled)")

    # -------------------------------------------------------------------------
    # VAL PASS — vectorised counter arrays; no score lists stored
    #
    # At k_val=100k the non-error score lists would be ~5B Python floats
    # (~140 GB).  Instead we compare against the cutoff immediately and
    # accumulate integer TP/FP/FN/TN counts per gene.
    # The only temporary is a (N_val, n_valid_genes) bool matrix (~1-2 GB).
    # -------------------------------------------------------------------------
    print("\nVal pass (accumulating TP/FP/FN/TN counters) ...")

    zeros = lambda: np.zeros(G, dtype=np.int64)
    tp_pool  = zeros(); fp_pool  = zeros()
    fn_pool  = zeros(); tn_pool  = zeros()
    nf_pool  = zeros(); nnf_pool = zeros()
    tp_var  = {v: zeros() for v in variations}
    fp_var  = {v: zeros() for v in variations}
    fn_var  = {v: zeros() for v in variations}
    tn_var  = {v: zeros() for v in variations}
    nf_var  = {v: zeros() for v in variations}
    nnf_var = {v: zeros() for v in variations}

    sign_val   = (X_val_all.astype(np.float32) * 2.0) - 1.0
    valid_pool = ~np.isnan(cutoff_vec_pool)

    for v in variations:
        print(f"    variation={v} ...")
        probs  = run_model_batched(X_val_all, num_not_shared=v)
        scores = _logit(probs) * sign_val
        del probs

        if valid_pool.any():
            s  = scores[:, valid_pool]
            cv = cutoff_vec_pool[valid_pool]
            m  = mask_val_all[:, valid_pool]
            det = s < cv[None, :]
            tp_pool[valid_pool]  += ( m  &  det).sum(0)
            fn_pool[valid_pool]  += ( m  & ~det).sum(0)
            fp_pool[valid_pool]  += (~m  &  det).sum(0)
            tn_pool[valid_pool]  += (~m  & ~det).sum(0)
            nf_pool[valid_pool]  +=   m.sum(0)
            nnf_pool[valid_pool] += (~m).sum(0)
            del s, cv, m, det

        valid_v = ~np.isnan(cutoff_vecs_var[v])
        if valid_v.any():
            s  = scores[:, valid_v]
            cv = cutoff_vecs_var[v][valid_v]
            m  = mask_val_all[:, valid_v]
            det = s < cv[None, :]
            tp_var[v][valid_v]  += ( m  &  det).sum(0)
            fn_var[v][valid_v]  += ( m  & ~det).sum(0)
            fp_var[v][valid_v]  += (~m  &  det).sum(0)
            tn_var[v][valid_v]  += (~m  & ~det).sum(0)
            nf_var[v][valid_v]  +=   m.sum(0)
            nnf_var[v][valid_v] += (~m).sum(0)
            del s, cv, m, det

        del scores

    del X_val_all, mask_val_all, sign_val
    print("\nAccumulation complete.")

    # -------------------------------------------------------------------------
    # Build DOR rows from counter arrays
    # -------------------------------------------------------------------------
    print("Computing per-gene DOR ...")

    cutoff_rows = []
    dor_rows    = []

    for gene_idx in range(G):
        cutoff = cutoff_vec_pool[gene_idx]
        if np.isnan(cutoff):
            continue

        n_tr = int(n_train_flipped_pool[gene_idx])
        nf   = int(nf_pool[gene_idx])
        nnf  = int(nnf_pool[gene_idx])
        gene_name, category = gene_cats.get(gene_idx, ('unknown', 'unknown'))
        cutoff_rows.append({
            'gene_idx': gene_idx, 'gene_name': gene_name, 'category': category,
            'n_train_flipped': n_tr, 'cutoff_99th': cutoff,
        })

        if nf < args.min_val_errors:
            continue
        stats = _dor_from_counts(tp_pool[gene_idx], fp_pool[gene_idx],
                                  fn_pool[gene_idx], tn_pool[gene_idx],
                                  nf, nnf, cutoff)
        dor_rows.append({'gene_idx': gene_idx, 'gene_name': gene_name,
                         'category': category, 'n_train_flipped': n_tr,
                         **stats})

    per_gene_cutoffs = pd.DataFrame(cutoff_rows)
    per_gene_dor = (
        pd.DataFrame(dor_rows)
        .sort_values('log10_dor', ascending=False, na_position='last')
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Per-variation DOR
    # -------------------------------------------------------------------------
    var_dor_rows = []

    for v in variations:
        for gene_idx in range(G):
            cutoff = cutoff_vecs_var[v][gene_idx]
            if np.isnan(cutoff):
                continue

            nf  = int(nf_var[v][gene_idx])
            nnf = int(nnf_var[v][gene_idx])
            if nf < args.min_val_errors:
                continue

            n_tr = int(n_train_flipped_var[v][gene_idx])
            gene_name, category = gene_cats.get(gene_idx, ('unknown', 'unknown'))
            stats = _dor_from_counts(tp_var[v][gene_idx], fp_var[v][gene_idx],
                                      fn_var[v][gene_idx], tn_var[v][gene_idx],
                                      nf, nnf, cutoff)
            var_dor_rows.append({'variation': v, 'gene_idx': gene_idx,
                                  'gene_name': gene_name, 'category': category,
                                  'n_train_flipped': n_tr, **stats})

    per_gene_dor_by_var = (
        pd.DataFrame(var_dor_rows)
        .sort_values(['variation', 'log10_dor'], ascending=[True, False], na_position='last')
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Category summary
    # -------------------------------------------------------------------------
    if not per_gene_dor.empty:
        cat_summary = per_gene_dor.groupby('category').agg(
            n_genes          =('gene_idx',    'count'),
            mean_log10_dor   =('log10_dor',   'mean'),
            median_log10_dor =('log10_dor',   'median'),
            mean_sensitivity =('sensitivity', 'mean'),
            mean_specificity =('specificity', 'mean'),
        ).reset_index()
    else:
        cat_summary = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    per_gene_dor.to_csv(        output_dir / 'per_gene_dor.tsv',              sep='\t', index=False)
    per_gene_dor_by_var.to_csv( output_dir / 'per_gene_dor_by_variation.tsv', sep='\t', index=False)
    per_gene_cutoffs.to_csv(    output_dir / 'per_gene_cutoffs.tsv',           sep='\t', index=False)
    if not cat_summary.empty:
        cat_summary.to_csv(     output_dir / 'category_summary.tsv',           sep='\t', index=False)

    # -------------------------------------------------------------------------
    # Summary text
    # -------------------------------------------------------------------------
    with open(output_dir / 'summary.txt', 'w') as fh:
        fh.write("Population Perturbation Experiment — Summary\n")
        fh.write("=" * 52 + "\n\n")

        fh.write("Genome split\n")
        fh.write(f"  Train (year < 2020): {n_train}  sampled: {k_train}  → cutoffs\n")
        fh.write(f"  Val   (year = 2020): {n_val}   sampled: {k_val}   → TP/FP/FN/TN\n")
        fh.write(f"  Test  (year > 2020): {n_test}  (never touched)\n\n")

        fh.write("Perturbation settings\n")
        fh.write(f"  Seeds:             {seeds}\n")
        fh.write(f"  Flip size:         {args.flip_size}\n")
        fh.write(f"  Variations:        {variations}\n")
        fh.write(f"  Mean flips/genome: {avg_flips:.1f}\n\n")

        fh.write(f"DOR analysis\n")
        fh.write(f"  min_errors_per_gene (train): {args.min_errors_per_gene}\n")
        fh.write(f"  min_val_errors:              {args.min_val_errors}\n")
        fh.write(f"  Genes with cutoff computed:  {len(per_gene_cutoffs)}\n")
        fh.write(f"  Genes with DOR reported:     {len(per_gene_dor)}\n")

        valid_dor = per_gene_dor['log10_dor'].dropna() if not per_gene_dor.empty else pd.Series([], dtype=float)
        if not valid_dor.empty:
            fh.write(f"  log10_dor  min:    {valid_dor.min():.3f}\n")
            fh.write(f"  log10_dor  median: {valid_dor.median():.3f}\n")
            fh.write(f"  log10_dor  mean:   {valid_dor.mean():.3f}\n")
            fh.write(f"  log10_dor  p90:    {float(np.percentile(valid_dor, 90)):.3f}\n")
            fh.write(f"  log10_dor  max:    {valid_dor.max():.3f}\n\n")

            fh.write("Top 20 genes by log10_dor\n")
            header = f"  {'gene_name':<45} {'category':<12} {'log10_dor':>10} {'sensitivity':>12} {'specificity':>12}\n"
            fh.write(header)
            fh.write("  " + "-" * (len(header) - 3) + "\n")
            for _, row in per_gene_dor.head(20).iterrows():
                dor_s  = f"{row['log10_dor']:.3f}"  if not np.isnan(row['log10_dor'])  else 'nan'
                sens_s = f"{row['sensitivity']:.3f}" if not np.isnan(row['sensitivity']) else 'nan'
                spec_s = f"{row['specificity']:.3f}" if not np.isnan(row['specificity']) else 'nan'
                fh.write(
                    f"  {str(row['gene_name']):<45} {str(row['category']):<12} "
                    f"{dor_s:>10} {sens_s:>12} {spec_s:>12}\n"
                )
            fh.write("\n")

        if not cat_summary.empty:
            fh.write("Category breakdown\n")
            for _, row in cat_summary.iterrows():
                fh.write(
                    f"  {row['category']:<12}: {int(row['n_genes'])} genes, "
                    f"mean_log10_dor={row['mean_log10_dor']:.3f}, "
                    f"mean_sensitivity={row['mean_sensitivity']:.3f}, "
                    f"mean_specificity={row['mean_specificity']:.3f}\n"
                )

    print(f"\nOutputs written to {output_dir}/")
    print(f"  per_gene_dor.tsv              — {len(per_gene_dor)} genes (pooled, train cutoffs / val eval)")
    print(f"  per_gene_dor_by_variation.tsv — {len(per_gene_dor_by_var)} rows ({len(variations)} variations)")
    print(f"  per_gene_cutoffs.tsv          — {len(per_gene_cutoffs)} genes (train cutoffs)")
    print(f"  category_summary.tsv          — {len(cat_summary)} categories")
    print(f"  train_val_test_split.tsv      — {len(split_df)} genomes")
    print(f"  summary.txt")


if __name__ == '__main__':
    main()
