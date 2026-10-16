#!/usr/bin/env python3
"""
plot_score_comparisons.py
==========================
Multiple visualization types for completeness and contamination scores at
variation=1, to compare which communicates results most clearly.

Reads:
    completeness_contamination_results/per_genome_scores.tsv

Outputs (in completeness_contamination_results/):
    violin_raw.png         violin_normalized.png
    histogram_raw.png      histogram_normalized.png
    boxwhisker_raw.png     boxwhisker_normalized.png

Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/plot_score_comparisons.py
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

PAIRS = [
    ('completeness_score',  'completeness_score_normalized',  'Completeness'),
    ('contamination_score', 'contamination_score_normalized', 'Contamination'),
]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def _style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


# ── panel renderers ───────────────────────────────────────────────────────────

def _violin_panel(ax, mags, refs, col, score_label, title):
    label_order = [LABEL_MAG, LABEL_REF]
    combo = pd.concat([
        mags[['genome_type', col]],
        refs[['genome_type', col]],
    ]).copy()
    combo['_g'] = combo['genome_type'].map({'MAG': LABEL_MAG, 'reference': LABEL_REF})

    sns.violinplot(
        data=combo, x='_g', y=col, hue='_g',
        order=label_order, hue_order=label_order, legend=False,
        palette={LABEL_MAG: C_MAG, LABEL_REF: C_REF},
        ax=ax, inner=None, linewidth=1.2, cut=2,
    )
    mag_plot = mags.copy(); mag_plot['_g'] = LABEL_MAG
    sns.stripplot(
        data=mag_plot, x='_g', y=col, hue='_g',
        order=label_order, hue_order=label_order, legend=False,
        palette={LABEL_MAG: C_MAG, LABEL_REF: C_REF},
        ax=ax, size=3.5, jitter=True, alpha=0.65, zorder=4,
    )
    ax.axhline(0, color='#888888', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(score_label, fontsize=10)
    _style_ax(ax)
    ax.spines['bottom'].set_visible(False)


def _histogram_panel(ax, mags, refs, col, score_label, title):
    mag_vals = mags[col].dropna().values
    ref_vals = refs[col].dropna().values
    bins = np.linspace(
        min(mag_vals.min(), ref_vals.min()),
        max(mag_vals.max(), ref_vals.max()), 60
    )
    ax.hist(ref_vals, bins=bins, density=True, alpha=0.6, color=C_REF, label=LABEL_REF)
    ax.hist(mag_vals, bins=bins, density=True, alpha=0.6, color=C_MAG, label=LABEL_MAG)
    ax.axvline(0, color='#888888', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(score_label, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    _style_ax(ax)


def _boxwhisker_panel(ax, mags, refs, col, score_label, title):
    mag_vals = mags[col].dropna().values
    ref_vals = refs[col].dropna().values

    def _bp(vals, pos, color):
        ax.boxplot(
            vals, positions=[pos], widths=0.4, showfliers=True,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color, linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor=color, markersize=2,
                            alpha=0.3, markeredgewidth=0, linestyle='none'),
            medianprops=dict(color='#1a1a1a', linewidth=2.0),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
        )

    _bp(ref_vals, 1, C_REF)
    _bp(mag_vals, 0, C_MAG)
    ax.axhline(0, color='#888888', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['MAG\n(n=143)', 'Reference\n(n=51,702)'], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(score_label, fontsize=10)
    _style_ax(ax)
    ax.spines['bottom'].set_visible(False)


# ── figure builder ────────────────────────────────────────────────────────────

def render_figure(mags, refs, col_comp, col_cont, score_label, panel_fn, out_name):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    panel_fn(ax_l, mags, refs, col_comp, score_label, 'Completeness')
    panel_fn(ax_r, mags, refs, col_cont, score_label, 'Contamination')

    legend_handles = [
        mpatches.Patch(facecolor=C_MAG, edgecolor='none', label='MAG assemblies (n=143)'),
        mpatches.Patch(facecolor=C_REF, edgecolor='none', label='Training-set references (n=51,702)'),
        Line2D([0], [0], color='#888888', linestyle='--', linewidth=1.2, label='Score = 0'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close(fig)


# ── sanity checks ─────────────────────────────────────────────────────────────

def print_sanity_checks(df):
    print('\n' + '=' * 70)
    print("SANITY CHECKS")
    print('=' * 70)

    # Cohen's d at each variation level
    print(f"\n--- Cohen's d (MAG vs Reference) ---")
    print(f"  {'Score':<40}  {'v=1':>6}  {'v=10':>6}  {'v=100':>6}")
    for col_raw, col_norm, label in PAIRS:
        for col, tag in [(col_raw, 'raw'), (col_norm, 'normalized')]:
            ds = []
            for V in VARS:
                sub = df[df['variation'] == V]
                ds.append(cohens_d(
                    sub[sub['genome_type'] == 'MAG'][col],
                    sub[sub['genome_type'] == 'reference'][col],
                ))
            name = f'{label} ({tag})'
            print(f"  {name:<40}  {ds[0]:>6.3f}  {ds[1]:>6.3f}  {ds[2]:>6.3f}")

    # Sign check at v=1
    print("\n--- Sign check (v=1) ---")
    v1   = df[df['variation'] == 1]
    mags = v1[v1['genome_type'] == 'MAG']
    refs = v1[v1['genome_type'] == 'reference']

    comp_ok = refs['completeness_score'].mean() < mags['completeness_score'].mean()
    cont_ok = refs['contamination_score'].mean() > mags['contamination_score'].mean()
    print(f"  Completeness  Ref={refs['completeness_score'].mean():.1f}  "
          f"MAG={mags['completeness_score'].mean():.1f}  "
          f"→ {'OK — refs more negative (more complete)' if comp_ok else 'REVERSED!'}")
    print(f"  Contamination Ref={refs['contamination_score'].mean():.1f}  "
          f"MAG={mags['contamination_score'].mean():.1f}  "
          f"→ {'OK — refs more positive (cleaner)' if cont_ok else 'REVERSED!'}")

    # IQR overlap
    print("\n--- Fraction of MAG scores within Reference IQR (v=1) ---")
    for col_raw, col_norm, label in PAIRS:
        for col, tag in [(col_raw, 'raw'), (col_norm, 'normalized')]:
            q1, q3 = refs[col].quantile(0.25), refs[col].quantile(0.75)
            n_in   = ((mags[col] >= q1) & (mags[col] <= q3)).sum()
            frac   = n_in / len(mags)
            print(f"  {label} ({tag}): {n_in}/{len(mags)} MAGs in Ref IQR "
                  f"[{q1:.2f}, {q3:.2f}]  ({frac:.1%})")

    # Top 5 most extreme completeness (least negative = most incomplete)
    print("\n--- Top 5 most incomplete MAGs by completeness score (v=1) ---")
    print("  (least negative score = model most surprised by absent genes)")
    top5 = mags.nlargest(5, 'completeness_score')[
        ['genome_id', 'completeness_score', 'n_absent_genes']
    ]
    for _, r in top5.iterrows():
        tag = '  ← bin assembly (SRR)' if 'SRR' in str(r['genome_id']) else ''
        print(f"  {r['genome_id'][:70]}{tag}")
        print(f"    completeness={r['completeness_score']:.1f}  "
              f"n_absent_genes={r['n_absent_genes']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(IN_FILE, sep='\t')
    print(f'Loaded {len(df):,} rows')

    v1   = df[df['variation'] == 1]
    mags = v1[v1['genome_type'] == 'MAG']
    refs = v1[v1['genome_type'] == 'reference']
    print(f'v=1: {len(mags)} MAGs, {len(refs):,} references')

    print('\nRendering violin plots ...')
    render_figure(mags, refs,
                  'completeness_score', 'contamination_score',
                  'Raw score', _violin_panel, 'violin_raw.png')
    render_figure(mags, refs,
                  'completeness_score_normalized', 'contamination_score_normalized',
                  'Normalized score (per gene)', _violin_panel, 'violin_normalized.png')

    print('\nRendering histograms ...')
    render_figure(mags, refs,
                  'completeness_score', 'contamination_score',
                  'Raw score', _histogram_panel, 'histogram_raw.png')
    render_figure(mags, refs,
                  'completeness_score_normalized', 'contamination_score_normalized',
                  'Normalized score (per gene)', _histogram_panel, 'histogram_normalized.png')

    print('\nRendering box-and-whisker plots ...')
    render_figure(mags, refs,
                  'completeness_score', 'contamination_score',
                  'Raw score', _boxwhisker_panel, 'boxwhisker_raw.png')
    render_figure(mags, refs,
                  'completeness_score_normalized', 'contamination_score_normalized',
                  'Normalized score (per gene)', _boxwhisker_panel, 'boxwhisker_normalized.png')

    print_sanity_checks(df)
    print('\nDone.')


if __name__ == '__main__':
    main()
