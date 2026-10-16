#!/usr/bin/env python3
"""
plot_score_overlap.py
======================
Single summary plot: pooled error vs non-error signed logit score distributions
across all genes, with three cutoff lines (A/B/C).

Density is estimated via histogram (O(n), no subsampling) so all training
genomes can be used.

Run from PATHB_sample root:
    conda run -n PathB python population_perturbation/plot_score_overlap.py [--k 999999] [--variation 100]

Output -> population_perturbation/distribution_diagnostics/score_overlap_v<variation>.png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE))

from population_perturbation import (
    load_vectors,
    parse_genome_filename,
    assign_split,
    _logit,
    run_model_batched,
    _build_batch,
)

G   = 5598


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      default='all_embeded_genomes/')
    parser.add_argument('--genes',     default='gene_names.txt')
    parser.add_argument('--k',         type=int, default=999999)
    parser.add_argument('--variation', type=int, default=100, choices=[1, 10, 100])
    parser.add_argument('--flip-size', type=int, default=20)
    parser.add_argument('--seeds',     default='42,123,7')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--output-dir', default='population_perturbation/distribution_diagnostics/')
    args = parser.parse_args()
    V = args.variation

    seeds = [int(s) for s in args.seeds.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load genomes
    # -------------------------------------------------------------------------
    data_dir    = Path(args.data)
    all_files   = sorted(data_dir.glob('*.txt'))
    train_paths = [p for p in all_files
                   if assign_split(*parse_genome_filename(p)) == 'train']
    rng     = random.Random(args.seed)
    k       = min(args.k, len(train_paths))
    sampled = rng.sample(train_paths, k)
    print(f'Using {k} train genomes (variation={V})')

    print('Loading vectors ...')
    X_clean = load_vectors(sampled)

    print('Building corrupted batch ...')
    X_corrupt, error_mask_all = _build_batch(X_clean, args.flip_size, seeds)
    n_seeds     = len(seeds)
    X_clean_rep = np.tile(X_clean, (n_seeds, 1))

    # Direction masks
    comp_error = error_mask_all & (X_clean_rep == 1)   # present→absent
    cont_error = error_mask_all & (X_clean_rep == 0)   # absent→present
    comp_ne    = (~error_mask_all) & (X_clean_rep == 0) # genuinely absent
    cont_ne    = (~error_mask_all) & (X_clean_rep == 1) # genuinely present

    print('Running model ...')
    probs  = run_model_batched(X_corrupt, num_not_shared=V)
    sign   = (X_corrupt.astype(np.float32) * 2.0) - 1.0
    scores = (_logit(probs) * sign).astype(np.float32)
    del probs, sign

    # -------------------------------------------------------------------------
    # Accumulate histograms without collecting full score arrays
    # (false_scores can be ~600M+ items with all train genomes)
    # -------------------------------------------------------------------------
    BIN_LO, BIN_HI, N_BINS = -15.0, 15.0, 600
    bin_edges  = np.linspace(BIN_LO, BIN_HI, N_BINS + 1)
    bin_width  = bin_edges[1] - bin_edges[0]
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    true_hist  = np.zeros(N_BINS, dtype=np.int64)
    false_hist = np.zeros(N_BINS, dtype=np.int64)

    for mask in (comp_error, cont_error):
        true_hist  += np.histogram(scores[mask], bins=bin_edges)[0]
    for mask in (comp_ne, cont_ne):
        false_hist += np.histogram(scores[mask], bins=bin_edges)[0]

    n_true  = int(true_hist.sum())
    n_false = int(false_hist.sum())
    print(f'True (error) samples:      {n_true:,}')
    print(f'False (non-error) samples: {n_false:,}')

    # Density (normalise so area = 1)
    y_true  = true_hist  / (n_true  * bin_width)
    y_false = false_hist / (n_false * bin_width)

    # -------------------------------------------------------------------------
    # Cutoffs via histogram CDFs — O(N_BINS), no per-sample sweep needed
    # -------------------------------------------------------------------------
    true_cdf  = np.cumsum(true_hist)  / n_true
    false_cdf = np.cumsum(false_hist) / n_false

    # A: 100% sensitivity — smallest upper bin edge where true_cdf >= 0.999
    idx_A    = int(np.searchsorted(true_cdf, 0.999))
    cutoff_A = float(bin_edges[min(idx_A + 1, N_BINS)])
    sens_A   = float(true_cdf[min(idx_A, N_BINS - 1)])
    spec_A   = float(1.0 - false_cdf[min(idx_A, N_BINS - 1)])

    # C: 100% specificity — largest upper bin edge where 1-false_cdf >= 0.999
    # i.e. false_cdf <= 0.001
    idx_C    = int(np.searchsorted(false_cdf, 0.001))
    cutoff_C = float(bin_edges[min(idx_C + 1, N_BINS)])
    sens_C   = float(true_cdf[min(idx_C, N_BINS - 1)])
    spec_C   = float(1.0 - false_cdf[min(idx_C, N_BINS - 1)])

    # B: Youden J = sens + spec - 1 = true_cdf - false_cdf (maximised per bin)
    j_vals   = true_cdf - false_cdf
    idx_B    = int(np.argmax(j_vals))
    cutoff_B = float(bin_edges[idx_B + 1])
    best_j   = float(j_vals[idx_B])
    sens_B   = float(true_cdf[idx_B])
    spec_B   = float(1.0 - false_cdf[idx_B])

    print(f'\nCutoff A (100% sensitivity): {cutoff_A:.3f}  '
          f'sens={sens_A:.4f}  spec={spec_A:.4f}')
    print(f'Cutoff B (optimal, J={best_j:.3f}): {cutoff_B:.3f}  '
          f'sens={sens_B:.4f}  spec={spec_B:.4f}')
    print(f'Cutoff C (100% specificity): {cutoff_C:.3f}  '
          f'sens={sens_C:.4f}  spec={spec_C:.4f}')

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')

    ax.fill_between(bin_centres, y_true,  alpha=0.18, color='#cc3333')
    ax.fill_between(bin_centres, y_false, alpha=0.18, color='#2a78d6')
    ax.plot(bin_centres, y_true,  color='#cc3333', linewidth=2.0,
            label=f'Error (flipped genes, n={n_true:,})')
    ax.plot(bin_centres, y_false, color='#2a78d6', linewidth=2.0,
            label=f'Non-error (correct genes, n={n_false:,})')

    # Cutoff lines
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax)           # lock y before adding text
    ymax = ax.get_ylim()[1]

    for label, c, color in [
        ('A', cutoff_A, '#e8a000'),
        ('B', cutoff_B, '#228b22'),
        ('C', cutoff_C, '#7b2fbe'),
    ]:
        ax.axvline(c, color=color, linewidth=1.8, linestyle='--', zorder=5)
        ax.text(c, ymax * 0.98, f' {label}', color=color,
                fontsize=10, fontweight='bold', va='top', ha='left')

    # Legend for cutoffs as a text block
    cutoff_text = (
        f'A  {cutoff_A:+.2f}  100% sensitivity  ({sens_A:.3f} / {spec_A:.3f})\n'
        f'B  {cutoff_B:+.2f}  optimal (Youden J={best_j:.3f})  ({sens_B:.3f} / {spec_B:.3f})\n'
        f'C  {cutoff_C:+.2f}  100% specificity  ({sens_C:.3f} / {spec_C:.3f})\n'
        f'                 (sensitivity / specificity)'
    )
    ax.text(0.02, 0.97, cutoff_text, transform=ax.transAxes,
            fontsize=8, va='top', ha='left',
            fontfamily='monospace',
            bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.85, boxstyle='round,pad=0.4'))

    ax.set_xlabel('Signed logit score  (lower = more error-like)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(
        f'Error vs Non-Error Score Distributions — variation={V}, k={k}, all genes pooled',
        fontsize=11, fontweight='bold'
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = output_dir / f'score_overlap_v{V}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
