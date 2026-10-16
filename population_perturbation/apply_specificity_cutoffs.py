#!/usr/bin/env python3
"""
apply_specificity_cutoffs.py
=============================
Generates per-gene completeness and contamination cutoffs using the
sensitivity-first algorithm from gene_cutoff_specificity.py, then compares
them to the original 99th-percentile cutoffs from per_gene_cutoffs.tsv.

No cached raw score files exist in this project — population_perturbation.py
only saves derived TSVs (cutoffs, DOR stats).  This script re-runs the model
on a sample of TRAIN-only genomes (year < 2020) to obtain per-gene score
distributions split by flip direction.

IMPORTANT: this script runs the PathB model and requires a GPU or sufficient
CPU.  It should NOT be run on the login node.  Submit via sbatch.

Run from PATHB_sample root:
    conda run -n PathB python population_perturbation/apply_specificity_cutoffs.py [options]

Sample sbatch invocation (pi_tami partition):
    sbatch --partition=pi_tami --time=4:00:00 --mem=32G --cpus-per-task=4 \\
           --wrap="conda run -n PathB python population_perturbation/apply_specificity_cutoffs.py --k 10000"

Outputs -> population_perturbation/specificity_cutoff_results/
    completeness_cutoffs.tsv
    contamination_cutoffs.tsv
    comparison_summary.txt
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

import sys
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE))

import run_model as _rm
from gene_cutoff_specificity import (
    compute_all_completeness_cutoffs,
    compute_all_contamination_cutoffs,
)
from population_perturbation import (
    load_gene_categories,
    load_vectors,
    parse_genome_filename,
    assign_split,
    _logit,
    run_model_batched,
    _build_batch,
)

G          = 5598
VARS       = [1, 10, 100]
BASELINES  = {V: float(np.log((1 - V / G) / (V / G))) for V in VARS}


def main():
    parser = argparse.ArgumentParser(
        description='Compute specificity-optimised per-gene cutoffs from train data.')
    parser.add_argument('--data',       default='all_embeded_genomes/',
                        help='Directory of genome .txt files')
    parser.add_argument('--genes',      default='gene_names.txt')
    parser.add_argument('--k',          type=int, default=5000,
                        help='Max train genomes to sample (default 5000; use 10000+ for production)')
    parser.add_argument('--flip-size',  type=int, default=20,
                        help='Genes flipped per genome per seed (must match original experiment)')
    parser.add_argument('--seeds',      default='42,123,7',
                        help='RNG seeds for flipping (must match original experiment)')
    parser.add_argument('--variations', default='1,10,100')
    parser.add_argument('--min-errors', type=int, default=50,
                        help='Min directional error instances per gene to compute cutoff')
    parser.add_argument('--min-sens',   type=float, default=0.999,
                        help='Required sensitivity (default 0.999 = 99.9%%)')
    parser.add_argument('--output-dir', default='population_perturbation/specificity_cutoff_results/')
    parser.add_argument('--seed',       type=int, default=42,
                        help='RNG seed for genome sampling')
    args = parser.parse_args()

    seeds      = [int(s) for s in args.seeds.split(',')]
    variations = [int(v) for v in args.variations.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading gene info ...')
    gene_cats  = load_gene_categories(args.genes)
    gene_names = [gene_cats[i][0] for i in range(G)]
    print(f'  {G} genes')

    # -------------------------------------------------------------------------
    # Discover and split genomes — TRAIN ONLY (year < 2020)
    # -------------------------------------------------------------------------
    data_dir    = Path(args.data)
    all_files   = sorted(data_dir.glob('*.txt'))
    train_paths = []
    for p in all_files:
        uy, cy = parse_genome_filename(p)
        if assign_split(uy, cy) == 'train':
            train_paths.append(p)

    print(f'Train genomes available: {len(train_paths)}  (year < 2020)')
    rng_sample = random.Random(args.seed)
    k          = min(args.k, len(train_paths))
    sampled    = rng_sample.sample(train_paths, k)
    print(f'Sampled: {k}')

    # -------------------------------------------------------------------------
    # Load clean genome vectors
    # -------------------------------------------------------------------------
    print(f'\nLoading {k} genome vectors ...')
    X_clean = load_vectors(sampled)
    print(f'  shape: {X_clean.shape}')

    # -------------------------------------------------------------------------
    # Build corrupted batch (all seeds stacked)
    # -------------------------------------------------------------------------
    print('Building corrupted batch ...')
    X_corrupt, error_mask_all = _build_batch(X_clean, args.flip_size, seeds)
    # X_corrupt   : (k * n_seeds, G) — corrupted input to model
    # error_mask_all : (k * n_seeds, G) — True where gene was flipped
    # X_clean must be tiled to match batch size
    n_seeds     = len(seeds)
    n_batch     = k * n_seeds
    X_clean_rep = np.tile(X_clean, (n_seeds, 1))  # (n_batch, G) — pre-flip values

    # Direction masks (derived from original presence, not corrupted)
    # present→absent flip: was 1 in clean, now flipped (0 in corrupt)
    comp_error_mask = error_mask_all & (X_clean_rep == 1)   # completeness errors
    # absent→present flip: was 0 in clean, now flipped (1 in corrupt)
    cont_error_mask = error_mask_all & (X_clean_rep == 0)   # contamination errors

    n_comp_errors = comp_error_mask.sum()
    n_cont_errors = cont_error_mask.sum()
    print(f'  Total completeness error instances (present→absent): {n_comp_errors:,}')
    print(f'  Total contamination error instances (absent→present): {n_cont_errors:,}')

    # -------------------------------------------------------------------------
    # Per-variation: compute signed logit scores, then cutoffs
    # -------------------------------------------------------------------------
    # Load existing 99th-pctile cutoffs for comparison
    old_cutoffs_path = _HERE / 'population_perturbation_results' / 'per_gene_cutoffs.tsv'
    old_cutoffs = None
    if old_cutoffs_path.exists():
        old_cutoffs = pd.read_csv(old_cutoffs_path, sep='\t').set_index('gene_idx')
        print(f'\nLoaded existing per_gene_cutoffs.tsv ({len(old_cutoffs)} genes)')
    else:
        print(f'\nWARN: {old_cutoffs_path} not found — comparison skipped')

    all_comp_dfs = {}
    all_cont_dfs = {}

    for V in variations:
        print(f'\n[variation = {V}]')
        print('  Running model ...')
        probs  = run_model_batched(X_corrupt, num_not_shared=V)
        sign   = (X_corrupt.astype(np.float32) * 2.0) - 1.0
        scores = (_logit(probs) * sign).astype(np.float32)
        del probs

        print('  Computing completeness cutoffs ...')
        comp_df = compute_all_completeness_cutoffs(
            scores, comp_error_mask, X_clean_rep,
            gene_names, args.min_sens, args.min_errors,
        )
        comp_df.insert(0, 'variation', V)

        print('  Computing contamination cutoffs ...')
        cont_df = compute_all_contamination_cutoffs(
            scores, cont_error_mask, X_clean_rep,
            gene_names, args.min_sens, args.min_errors,
        )
        cont_df.insert(0, 'variation', V)

        all_comp_dfs[V] = comp_df
        all_cont_dfs[V] = cont_df

        # Quick per-variation summary
        for label, df in [('Completeness', comp_df), ('Contamination', cont_df)]:
            valid  = df.dropna(subset=['cutoff'])
            n_s1   = (valid['step_used'] == 1).sum()
            n_s2   = (valid['step_used'] == 2).sum()
            n_s3   = (valid['step_used'] == 3).sum()
            n_s4   = (valid['step_used'] == 4).sum()
            n_met  = valid['sensitivity_requirement_met'].sum()
            print(f'  {label}: {len(valid)} genes with cutoff  '
                  f'step1={n_s1} step2={n_s2} step3={n_s3} step4={n_s4}  '
                  f'sens_met={n_met}')

        del scores, sign

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    comp_out = pd.concat(all_comp_dfs.values(), ignore_index=True)
    cont_out = pd.concat(all_cont_dfs.values(), ignore_index=True)

    comp_path = output_dir / 'completeness_cutoffs.tsv'
    cont_path = output_dir / 'contamination_cutoffs.tsv'
    comp_out.to_csv(comp_path, sep='\t', index=False, float_format='%.6f')
    cont_out.to_csv(cont_path, sep='\t', index=False, float_format='%.6f')
    print(f'\nSaved: {comp_path}  ({len(comp_out):,} rows)')
    print(f'Saved: {cont_path}  ({len(cont_out):,} rows)')

    # -------------------------------------------------------------------------
    # Comparison summary
    # -------------------------------------------------------------------------
    lines = [
        'Specificity Cutoff Results — Comparison Summary',
        '=' * 52,
        f'k_sampled={k}  flip_size={args.flip_size}  seeds={seeds}',
        f'min_sensitivity={args.min_sens}  min_errors={args.min_errors}',
        '',
    ]

    for V in variations:
        comp_df = all_comp_dfs[V].dropna(subset=['cutoff'])
        cont_df = all_cont_dfs[V].dropna(subset=['cutoff'])
        lines.append(f'=== variation = {V} ===')

        if old_cutoffs is not None:
            old_v = old_cutoffs.reindex(range(G))['cutoff_99th'].dropna()
            lines.append(f'  Old 99th-pctile cutoff   : mean={old_v.mean():.3f}  '
                         f'median={old_v.median():.3f}  n={len(old_v)}')

        for label, df in [('Completeness (new)', comp_df), ('Contamination (new)', cont_df)]:
            c = df['cutoff']
            s_med  = df['sensitivity'].median()
            sp_med = df['specificity'].median()
            n_s1   = (df['step_used'] == 1).sum()
            n_s3   = (df['step_used'] == 3).sum()
            n_met  = df['sensitivity_requirement_met'].sum()
            lines.append(
                f'  {label:<26}: mean={c.mean():.3f}  median={c.median():.3f}  '
                f'n={len(df)}'
            )
            lines.append(
                f'    sens med={s_med:.4f}  spec med={sp_med:.4f}  '
                f'step1={n_s1}  step3={n_s3}  sens_met={n_met}/{len(df)}'
            )

        lines.append('')

    # Sensitivity distribution
    lines.append('--- Sensitivity distribution across all variations ---')
    for V in variations:
        for label, df in [('comp', all_comp_dfs[V]), ('cont', all_cont_dfs[V])]:
            valid = df['sensitivity'].dropna()
            pcts  = [50, 90, 99, 100]
            vals  = np.nanpercentile(valid, pcts)
            lines.append(
                f'  v={V} {label}: ' +
                '  '.join(f'p{p}={v:.4f}' for p, v in zip(pcts, vals))
            )

    summary_path = output_dir / 'comparison_summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n')
    print(f'Saved: {summary_path}')

    # Print to console too
    print('\n' + '\n'.join(lines))
    print('\nDone.')


if __name__ == '__main__':
    main()
