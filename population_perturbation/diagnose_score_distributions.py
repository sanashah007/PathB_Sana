#!/usr/bin/env python3
"""
diagnose_score_distributions.py
================================
Diagnostic script: characterises error vs non-error signed logit score
distributions per gene to understand why per-gene cutoffs show full overlap.

Uses the same genome vectors, model, and flip logic as apply_specificity_cutoffs.py.
Only variation=100 is run (most informative for overlap analysis).

Scoring convention (same as gene_cutoff_specificity.py):
    score = logit(prob) × sign    where sign = 2×presence − 1
    Lower score → more error-like.

Run from PATHB_sample root:
    conda run -n PathB python population_perturbation/diagnose_score_distributions.py [--k 5000]

Outputs -> population_perturbation/distribution_diagnostics/
    per_gene_diagnostics.tsv
    summary.txt
    distribution_plots.png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE))

from population_perturbation import (
    load_gene_categories,
    load_vectors,
    parse_genome_filename,
    assign_split,
    _logit,
    run_model_batched,
    _build_batch,
)

G    = 5598
V    = 100   # variation — most informative for overlap


# ---------------------------------------------------------------------------
# Per-gene metrics
# ---------------------------------------------------------------------------

def _pooled_std(s1, n1, s2, n2):
    if n1 + n2 <= 2:
        return np.nan
    return np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))


def compute_gene_metrics(true_scores, false_scores):
    """
    Compute all diagnostic metrics for one gene (one score type).
    true_scores : 1-D array of signed logit scores where gene was flipped.
    false_scores: 1-D array where gene was NOT flipped and correctly present/absent.
    """
    n_t = len(true_scores)
    n_f = len(false_scores)

    if n_t == 0:
        return dict(
            true_n=0, true_mean=np.nan, true_std=np.nan,
            true_min=np.nan, true_max=np.nan,
            false_n=n_f, false_mean=np.nan, false_std=np.nan,
            false_min=np.nan, false_max=np.nan,
            overlap_fraction=np.nan, cohens_d=np.nan,
            achievable_specificity_at_100pct_sensitivity=np.nan,
            achievable_sensitivity_at_100pct_specificity=np.nan,
        )

    t_mean = float(np.mean(true_scores))
    t_std  = float(np.std(true_scores,  ddof=1)) if n_t > 1 else 0.0
    t_min  = float(np.min(true_scores))
    t_max  = float(np.max(true_scores))

    if n_f == 0:
        return dict(
            true_n=n_t, true_mean=t_mean, true_std=t_std,
            true_min=t_min, true_max=t_max,
            false_n=0, false_mean=np.nan, false_std=np.nan,
            false_min=np.nan, false_max=np.nan,
            overlap_fraction=np.nan, cohens_d=np.nan,
            achievable_specificity_at_100pct_sensitivity=np.nan,
            achievable_sensitivity_at_100pct_specificity=np.nan,
        )

    f_mean = float(np.mean(false_scores))
    f_std  = float(np.std(false_scores, ddof=1)) if n_f > 1 else 0.0
    f_min  = float(np.min(false_scores))
    f_max  = float(np.max(false_scores))

    # Cohen's d: (false_mean - true_mean) / pooled_std
    # Positive when false scores are higher than true scores (expected for a
    # separable gene since lower = more error-like)
    ps  = _pooled_std(t_std, n_t, f_std, n_f)
    if ps > 0:
        cohens_d = (f_mean - t_mean) / ps
    else:
        cohens_d = np.nan

    # Fraction of true samples that fall within [false_min, false_max]
    overlap_fraction = float(((true_scores >= f_min) & (true_scores <= f_max)).mean())

    # Achievable specificity at 100% sensitivity: cutoff = max(true), spec = P(false > max(true))
    achievable_spec = float((false_scores > t_max).mean())

    # Achievable sensitivity at 100% specificity: cutoff just below min(false),
    # i.e. sensitivity = P(true < min(false))
    achievable_sens = float((true_scores < f_min).mean())

    return dict(
        true_n=n_t,  true_mean=t_mean, true_std=t_std,  true_min=t_min,  true_max=t_max,
        false_n=n_f, false_mean=f_mean, false_std=f_std, false_min=f_min, false_max=f_max,
        overlap_fraction=overlap_fraction,
        cohens_d=cohens_d,
        achievable_specificity_at_100pct_sensitivity=achievable_spec,
        achievable_sensitivity_at_100pct_specificity=achievable_sens,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Diagnose per-gene score distributions.')
    parser.add_argument('--data',      default='all_embeded_genomes/',
                        help='Directory of genome .txt files')
    parser.add_argument('--genes',     default='gene_names.txt')
    parser.add_argument('--k',         type=int, default=5000,
                        help='Max train genomes to use (default 5000; 1000 is fine for quick diagnostics)')
    parser.add_argument('--flip-size', type=int, default=20)
    parser.add_argument('--seeds',     default='42,123,7')
    parser.add_argument('--output-dir', default='population_perturbation/distribution_diagnostics/')
    parser.add_argument('--seed',      type=int, default=42, help='Sampling RNG seed')
    args = parser.parse_args()

    seeds      = [int(s) for s in args.seeds.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Gene info
    # -------------------------------------------------------------------------
    print('Loading gene info ...')
    gene_cats  = load_gene_categories(args.genes)
    gene_names = [gene_cats[i][0] for i in range(G)]
    gene_cats_ = [gene_cats[i][1] for i in range(G)]

    # -------------------------------------------------------------------------
    # Train genomes
    # -------------------------------------------------------------------------
    data_dir    = Path(args.data)
    all_files   = sorted(data_dir.glob('*.txt'))
    train_paths = [p for p in all_files
                   if assign_split(*parse_genome_filename(p)) == 'train']
    print(f'Train genomes available: {len(train_paths)}  (year < 2020)')
    rng    = random.Random(args.seed)
    k      = min(args.k, len(train_paths))
    sampled = rng.sample(train_paths, k)
    print(f'Using k={k}')

    # -------------------------------------------------------------------------
    # Load vectors & build corrupted batch
    # -------------------------------------------------------------------------
    print(f'Loading {k} genome vectors ...')
    X_clean = load_vectors(sampled)
    print(f'  shape: {X_clean.shape}')

    print('Building corrupted batch ...')
    X_corrupt, error_mask_all = _build_batch(X_clean, args.flip_size, seeds)
    n_seeds     = len(seeds)
    n_batch     = k * n_seeds
    X_clean_rep = np.tile(X_clean, (n_seeds, 1))

    comp_error_mask = error_mask_all & (X_clean_rep == 1)
    cont_error_mask = error_mask_all & (X_clean_rep == 0)

    print(f'  Batch shape: {X_corrupt.shape}')
    print(f'  Completeness error instances: {comp_error_mask.sum():,}')
    print(f'  Contamination error instances: {cont_error_mask.sum():,}')

    # -------------------------------------------------------------------------
    # Run model (variation=100 only)
    # -------------------------------------------------------------------------
    print(f'\nRunning model (variation={V}) ...')
    probs  = run_model_batched(X_corrupt, num_not_shared=V)
    sign   = (X_corrupt.astype(np.float32) * 2.0) - 1.0
    scores = (_logit(probs) * sign).astype(np.float32)
    del probs

    # -------------------------------------------------------------------------
    # Non-error selectors (match gene_cutoff_specificity.py exactly)
    # -------------------------------------------------------------------------
    comp_ne_mask = (~error_mask_all) & (X_clean_rep == 0)  # genuinely absent
    cont_ne_mask = (~error_mask_all) & (X_clean_rep == 1)  # genuinely present

    # -------------------------------------------------------------------------
    # Per-gene metrics
    # -------------------------------------------------------------------------
    print('Computing per-gene diagnostics ...')
    rows = []
    for g in range(G):
        for score_type, err_mask, ne_mask in [
            ('completeness', comp_error_mask[:, g], comp_ne_mask[:, g]),
            ('contamination', cont_error_mask[:, g], cont_ne_mask[:, g]),
        ]:
            t = scores[err_mask, g]
            f = scores[ne_mask,  g]
            m = compute_gene_metrics(t, f)
            row = {
                'gene_name':  gene_names[g],
                'gene_idx':   g,
                'category':   gene_cats_[g],
                'score_type': score_type,
            }
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    comp_df = df[df.score_type == 'completeness'].copy()
    cont_df = df[df.score_type == 'contamination'].copy()

    for label, sub in [('Completeness', comp_df), ('Contamination', cont_df)]:
        valid = sub.dropna(subset=['true_n', 'false_n'])
        valid = valid[valid.true_n > 0]
        print(f'\n=== {label} ===')
        print(f'  Median true_n  (samples per gene): {valid.true_n.median():.0f}')
        print(f'  Median false_n (samples per gene): {valid.false_n.median():.0f}')
        # Genes with any cutoff giving sens>0.5 and spec>0.5
        has_both = valid[
            (valid.achievable_specificity_at_100pct_sensitivity > 0.5) |
            (valid.achievable_sensitivity_at_100pct_specificity > 0.5)
        ]
        print(f'  Genes with any cutoff achieving both sens>0.5 and spec>0.5: '
              f'{len(has_both)}/{len(valid)} ({100*len(has_both)/max(1,len(valid)):.1f}%)')

    # 5 example full-overlap genes (completeness, lowest achievable_spec)
    ex = comp_df.dropna(subset=['achievable_specificity_at_100pct_sensitivity'])
    ex = ex[ex.true_n > 0].nsmallest(5, 'achievable_specificity_at_100pct_sensitivity')
    print('\n--- 5 most-overlapping completeness genes ---')
    for _, row in ex.iterrows():
        print(f'  {row.gene_name[:45]:<45}  true=[{row.true_min:.2f},{row.true_max:.2f}] '
              f'false=[{row.false_min:.2f},{row.false_max:.2f}]  '
              f'd={row.cohens_d:.3f}  overlap={row.overlap_fraction:.3f}')

    # -------------------------------------------------------------------------
    # Save TSV
    # -------------------------------------------------------------------------
    tsv_path = output_dir / 'per_gene_diagnostics.tsv'
    df.to_csv(tsv_path, sep='\t', index=False, float_format='%.5f')
    print(f'\nSaved: {tsv_path}  ({len(df):,} rows)')

    # -------------------------------------------------------------------------
    # Summary text
    # -------------------------------------------------------------------------
    lines = ['Score Distribution Diagnostics', '=' * 52,
             f'k={k}  flip_size={args.flip_size}  seeds={seeds}  variation={V}', '']

    for label, sub in [('Completeness (present→absent)', comp_df),
                        ('Contamination (absent→present)', cont_df)]:
        valid = sub.dropna(subset=['cohens_d'])
        valid = valid[valid.true_n > 0]
        lines.append(f'--- {label} ---')
        lines.append(f'  Genes analysed: {len(valid)}')

        pcts = [25, 50, 75, 90, 99]
        cd   = valid.cohens_d.dropna()
        pct_vals = np.percentile(cd, pcts)
        lines.append("  Cohen's d distribution: " +
                     '  '.join(f'p{p}={v:.3f}' for p, v in zip(pcts, pct_vals)))
        lines.append(f"  Genes with d > 1 (clearly separable): {(cd > 1).sum()}")
        lines.append(f"  Genes with d < 0.2 (no separation):   {(cd < 0.2).sum()}")

        of = valid.overlap_fraction.dropna()
        lines.append(f'  Mean overlap_fraction: {of.mean():.4f}  '
                     f'(p50={np.percentile(of, 50):.4f}  p90={np.percentile(of, 90):.4f})')

        asp = valid.achievable_specificity_at_100pct_sensitivity.dropna()
        lines.append(f'  Mean achievable specificity @ 100% sensitivity: {asp.mean():.4f}  '
                     f'(p50={np.percentile(asp, 50):.4f}  p90={np.percentile(asp, 90):.4f})')

        asn = valid.achievable_sensitivity_at_100pct_specificity.dropna()
        lines.append(f'  Mean achievable sensitivity @ 100% specificity: {asn.mean():.4f}  '
                     f'(p50={np.percentile(asn, 50):.4f}  p90={np.percentile(asn, 90):.4f})')

        # Top 20 by Cohen's d
        top20 = valid.nlargest(20, 'cohens_d')
        lines.append('\n  Top 20 genes by Cohen\'s d (most separable):')
        lines.append(f'  {"gene_name":<45} {"category":<12} {"d":>6} '
                     f'{"true_mean":>10} {"false_mean":>10} {"overlap":>8}')
        for _, r in top20.iterrows():
            lines.append(f'  {str(r.gene_name)[:45]:<45} {str(r.category):<12} '
                         f'{r.cohens_d:6.3f} {r.true_mean:10.3f} {r.false_mean:10.3f} '
                         f'{r.overlap_fraction:8.4f}')

        # Bottom 20 by Cohen's d
        bot20 = valid[valid.true_n >= 10].nsmallest(20, 'cohens_d')
        lines.append('\n  Bottom 20 genes by Cohen\'s d (least separable, n≥10):')
        lines.append(f'  {"gene_name":<45} {"category":<12} {"d":>6} '
                     f'{"true_n":>6} {"false_n":>6} {"overlap":>8}')
        for _, r in bot20.iterrows():
            lines.append(f'  {str(r.gene_name)[:45]:<45} {str(r.category):<12} '
                         f'{r.cohens_d:6.3f} {int(r.true_n):6d} {int(r.false_n):6d} '
                         f'{r.overlap_fraction:8.4f}')

        # Category breakdown
        lines.append('\n  Category breakdown (median Cohen\'s d):')
        cat_grp = valid.groupby('category')['cohens_d'].agg(['median', 'count'])
        for cat, row in cat_grp.sort_values('median', ascending=False).iterrows():
            lines.append(f'    {cat:<14}: n={int(row["count"]):5d}  median d={row["median"]:.3f}')

        lines.append('')

    summary_path = output_dir / 'summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n')
    print(f'Saved: {summary_path}')
    print('\n' + '\n'.join(lines))

    # -------------------------------------------------------------------------
    # Distribution plots (2×3 grid, completeness genes)
    # -------------------------------------------------------------------------
    print('\nBuilding distribution plots ...')

    valid_comp = comp_df.dropna(subset=['cohens_d'])
    valid_comp = valid_comp[valid_comp.true_n >= 5].copy()

    # Row 1: top 3 by Cohen's d
    top3  = valid_comp.nlargest(3, 'cohens_d')
    # Row 2: 3 near-median Cohen's d
    med_d = valid_comp.cohens_d.median()
    valid_comp['_dist'] = (valid_comp.cohens_d - med_d).abs()
    mid3  = valid_comp.nsmallest(3, '_dist')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor('white')

    row_labels = ['Top 3 most separable (highest Cohen\'s d)',
                  '3 typical genes (median Cohen\'s d)']

    for row_idx, (row_label, gene_rows) in enumerate([(row_labels[0], top3),
                                                       (row_labels[1], mid3)]):
        for col_idx, (_, gr) in enumerate(gene_rows.reset_index().iterrows()):
            ax = axes[row_idx][col_idx]
            g  = int(gr.gene_idx)

            t_scores = scores[comp_error_mask[:, g], g]
            f_scores = scores[comp_ne_mask[:, g],    g]

            if len(t_scores) == 0 or len(f_scores) == 0:
                ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            lo = min(t_scores.min(), f_scores.min())
            hi = max(t_scores.max(), f_scores.max())
            bins = np.linspace(lo, hi, 50)

            ax.hist(f_scores, bins=bins, density=True, color='#2a78d6', alpha=0.5,
                    label=f'non-error (n={len(f_scores):,})')
            ax.hist(t_scores, bins=bins, density=True, color='#cc3333', alpha=0.6,
                    label=f'error (n={len(t_scores):,})')

            ymax = ax.get_ylim()[1] or 1.0
            ax.axvline(float(gr.true_mean),  color='#cc3333', linewidth=1.8,
                       linestyle='--', label=f'error mean={gr.true_mean:.2f}')
            ax.axvline(float(gr.false_mean), color='#2a78d6', linewidth=1.8,
                       linestyle='--', label=f'non-error mean={gr.false_mean:.2f}')

            gname = str(gr.gene_name)
            short = gname[:35] + '…' if len(gname) > 35 else gname
            ax.set_title(f'{short}\nd={gr.cohens_d:.3f}  overlap={gr.overlap_fraction:.3f}',
                         fontsize=8, fontweight='bold')
            ax.set_xlabel('signed logit score', fontsize=8)
            ax.set_ylabel('density', fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7)
            if col_idx == 0:
                ax.legend(fontsize=6, frameon=False)

        # Row label
        axes[row_idx][0].annotate(row_label, xy=(-0.15, 0.5),
                                  xycoords='axes fraction', fontsize=8,
                                  rotation=90, va='center', ha='right',
                                  fontweight='bold')

    fig.suptitle(f'Completeness: Error vs Non-Error Score Distributions (variation={V}, k={k})',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()

    plot_path = output_dir / 'distribution_plots.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {plot_path}')
    plt.close(fig)

    print('\nDone.')


if __name__ == '__main__':
    main()
