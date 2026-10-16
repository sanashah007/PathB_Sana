#!/usr/bin/env python3
"""
plot_completeness_contamination.py
===================================
Violin plots comparing completeness and contamination scores between MAG
assemblies and training-set reference genomes.

Reads:
    completeness_contamination_results/per_genome_scores.tsv

Outputs (in completeness_contamination_results/):
    completeness_contamination_raw.png/.pdf
    completeness_contamination_normalized.png/.pdf

Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/plot_completeness_contamination.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

_HERE   = Path(__file__).resolve().parent
IN_FILE = _HERE / 'completeness_contamination_results' / 'per_genome_scores.tsv'
OUT_DIR = _HERE / 'completeness_contamination_results'

C_MAG = '#eb6834'
C_REF = '#2a78d6'
VARS  = [1, 10, 100]
G     = 5598
BASELINES = {V: np.log((1 - V / G) / (V / G)) for V in VARS}

LABEL_MAG = 'MAG (n=143)'
LABEL_REF = 'Reference (n=51,702)'

SCORE_PAIRS = [
    ('completeness_score',  'completeness_score_normalized',  'Completeness'),
    ('contamination_score', 'contamination_score_normalized', 'Contamination'),
]


def cohens_d(a, b):
    """Absolute pooled-SD Cohen's d."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def _style_ax(ax):
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=8)


def render_figure(df, col_comp, col_cont, ylabel, out_stem):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    label_order = [LABEL_MAG, LABEL_REF]

    for row_idx, (col, row_label) in enumerate(
        [(col_comp, 'Completeness'), (col_cont, 'Contamination')]
    ):
        for col_idx, V in enumerate(VARS):
            ax  = axes[row_idx, col_idx]
            sub = df[df['variation'] == V].copy()
            sub['_g'] = sub['genome_type'].map({'MAG': LABEL_MAG, 'reference': LABEL_REF})

            sns.violinplot(
                data=sub, x='_g', y=col, hue='_g',
                order=label_order, hue_order=label_order, legend=False,
                palette={LABEL_MAG: C_MAG, LABEL_REF: C_REF},
                ax=ax, inner=None, linewidth=1.2, cut=2,
            )

            sns.stripplot(
                data=sub[sub['genome_type'] == 'MAG'],
                x='_g', y=col, hue='_g',
                order=label_order, hue_order=label_order, legend=False,
                palette={LABEL_MAG: C_MAG, LABEL_REF: C_REF},
                ax=ax, size=3.5, jitter=True, alpha=0.65, zorder=4,
            )

            # mean diamonds
            for gtype, x_pos in [('MAG', 0), ('reference', 1)]:
                mean_val = sub[sub['genome_type'] == gtype][col].mean()
                ax.scatter(x_pos, mean_val, color='white', s=55, zorder=6,
                           marker='D', edgecolors='#333333', linewidths=1.0)

            ax.axhline(0, color='#888888', linestyle='--', linewidth=0.85, alpha=0.8, zorder=2)

            ax.set_title(f'v={V}  (baseline = {BASELINES[V]:.2f})',
                         fontsize=10, fontweight='bold', pad=6)
            ax.set_xlabel('')
            ax.set_ylabel(f'{row_label}\n{ylabel}' if col_idx == 0 else '', fontsize=10)
            _style_ax(ax)

    legend_handles = [
        mpatches.Patch(facecolor=C_MAG, edgecolor='none', label='MAG assemblies (n=143)'),
        mpatches.Patch(facecolor=C_REF, edgecolor='none', label='Training-set references (n=51,702)'),
        Line2D([0], [0], color='#888888', linestyle='--', linewidth=1.2, label='y = 0'),
        Line2D([0], [0], linestyle='none', marker='D', markersize=7,
               markerfacecolor='white', markeredgecolor='#333333',
               markeredgewidth=1.0, label='Mean'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext, kw in [('png', {'dpi': 300}), ('pdf', {})]:
        out = OUT_DIR / f'{out_stem}.{ext}'
        fig.savefig(out, bbox_inches='tight', **kw)
        print(f'  Saved: {out}')
    plt.close(fig)


def print_stats(df):
    print('\n' + '=' * 70)
    print('DESCRIPTIVE STATISTICS')
    print('=' * 70)

    for col_raw, col_norm, label in SCORE_PAIRS:
        print(f'\n--- {label} ---')
        for V in VARS:
            sub  = df[df['variation'] == V]
            mags = sub[sub['genome_type'] == 'MAG']
            refs = sub[sub['genome_type'] == 'reference']
            print(f'\n  v={V}:')
            for col, tag in [(col_raw, 'raw'), (col_norm, 'normalized')]:
                d = cohens_d(mags[col], refs[col])
                print(f'    [{tag:>10}]  MAG mean={mags[col].mean():+10.3f}  median={mags[col].median():+10.3f}'
                      f'   Ref mean={refs[col].mean():+10.3f}  median={refs[col].median():+10.3f}'
                      f"   Cohen's d={d:.3f}")

    print('\n' + '=' * 70)
    print("SEPARATION SUMMARY (Cohen's d — higher = cleaner MAG vs Ref separation)")
    print('=' * 70)
    print(f"{'Score':<25}" + ''.join(f'  v={V} raw   v={V} norm' for V in VARS))

    all_d = []
    for col_raw, col_norm, label in SCORE_PAIRS:
        row = f'{label:<25}'
        for V in VARS:
            sub  = df[df['variation'] == V]
            mags = sub[sub['genome_type'] == 'MAG']
            refs = sub[sub['genome_type'] == 'reference']
            d_raw  = cohens_d(mags[col_raw],  refs[col_raw])
            d_norm = cohens_d(mags[col_norm], refs[col_norm])
            all_d.append(('raw',  label, V, d_raw))
            all_d.append(('norm', label, V, d_norm))
            row += f'  {d_raw:>7.3f}   {d_norm:>8.3f}'
        print(row)

    best      = max(all_d, key=lambda x: x[3])
    avg_raw   = np.mean([d for tag, _, _, d in all_d if tag == 'raw'])
    avg_norm  = np.mean([d for tag, _, _, d in all_d if tag == 'norm'])
    winner    = 'normalized' if avg_norm > avg_raw else 'raw'
    print(f"\nBest separation: {best[1]} {'normalized' if best[0]=='norm' else 'raw'}"
          f" at v={best[2]} (d={best[3]:.3f})")
    print(f"Mean d — raw: {avg_raw:.3f}  normalized: {avg_norm:.3f}"
          f"  → {winner} scores show cleaner separation on average")


def main():
    df = pd.read_csv(IN_FILE, sep='\t')
    print(f'Loaded {len(df):,} rows — {df["genome_type"].value_counts().to_dict()}')

    print('\nRendering raw score figure ...')
    render_figure(df,
                  col_comp='completeness_score',
                  col_cont='contamination_score',
                  ylabel='Raw score',
                  out_stem='completeness_contamination_raw')

    print('\nRendering normalized score figure ...')
    render_figure(df,
                  col_comp='completeness_score_normalized',
                  col_cont='contamination_score_normalized',
                  ylabel='Normalized score (per gene)',
                  out_stem='completeness_contamination_normalized')

    print_stats(df)
    print('\nDone.')


if __name__ == '__main__':
    main()
