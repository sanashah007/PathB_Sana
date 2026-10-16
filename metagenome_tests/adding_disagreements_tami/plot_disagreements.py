#!/usr/bin/env python3
"""
plot_disagreements.py
=====================
Box-and-whisker plots of raw PathB disagreement counts: MAG assemblies vs
training-set references, one panel per variation level.

Reads:
    disagreement_distribution_results/per_genome_summary.tsv

Output:
    disagreement_plots.png

Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/adding_disagreements_tami/plot_disagreements.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_HERE       = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / 'disagreement_distribution_results'

C_MAG = '#2a78d6'
C_REF = '#eb6834'
VARS  = [1, 10, 100]


def main():
    pg  = pd.read_csv(RESULTS_DIR / 'per_genome_summary.tsv', sep='\t')
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        sub  = pg[pg['variation'] == V]
        mags = sub[sub['genome_type'] == 'MAG']['n_disagreements'].values
        refs = sub[sub['genome_type'] == 'reference']['n_disagreements'].values

        def _bp(vals, pos, color):
            ax.boxplot(
                vals, positions=[pos], widths=0.4,
                showfliers=True, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.6, edgecolor=color, linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor=color, markersize=2,
                                alpha=0.25, markeredgewidth=0, linestyle='none'),
                medianprops=dict(color='#1a1a1a', linewidth=2.0),
                whiskerprops=dict(color=color, linewidth=1.5),
                capprops=dict(color=color, linewidth=1.5),
            )

        _bp(refs, 1, C_REF)
        _bp(mags, 0, C_MAG)

        # individual MAG points
        jitter = rng.uniform(-0.15, 0.15, len(mags))
        ax.scatter(jitter, mags, color=C_MAG, alpha=0.7, s=20, zorder=5, linewidths=0)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'MAG\n(n={len(mags)})', f'Ref\n(n={len(refs):,})'], fontsize=9)
        ax.set_title(f'variation = {V}', fontweight='bold', fontsize=11)
        ax.set_ylabel('Disagreements per genome' if V == 1 else '', fontsize=10)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=9)

        for pos, vals, c in [(0, mags, C_MAG), (1, refs, C_REF)]:
            ax.text(pos + 0.22, np.median(vals),
                    f'med={np.median(vals):.0f}', fontsize=7.5,
                    color=c, fontweight='bold', va='center')

    legend_handles = [
        mpatches.Patch(facecolor=C_MAG, edgecolor='none', label='MAG assemblies'),
        mpatches.Patch(facecolor=C_REF, edgecolor='none', label='Training-set references'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('PathB Raw Disagreements (logit contradicts input): MAGs vs References',
                 fontsize=12, fontweight='bold', y=1.06)

    plt.tight_layout()
    out = _HERE / 'disagreement_plots.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
