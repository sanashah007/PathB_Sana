#!/usr/bin/env python3
"""
plot_specificity_cutoffs.py
============================
Visualises the per-gene specificity cutoffs produced by apply_specificity_cutoffs.py.

Three figures:
  1. cutoff_distributions.png  — overlapping histograms: old 99th-pctile vs new
                                  completeness vs new contamination cutoffs, 1×3 by variation
  2. specificity_distributions.png — achieved specificity per gene (comp vs cont), 1×3
  3. step_usage.png             — bar chart of Step 1 / Step 2 / Step 3 counts, 1×3

Run from PATHB_sample root:
    conda run -n PathB python population_perturbation/plot_specificity_cutoffs.py

Outputs -> population_perturbation/specificity_cutoff_results/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
RES_DIR  = _HERE / 'specificity_cutoff_results'
OLD_TSV  = _HERE / 'population_perturbation_results' / 'per_gene_cutoffs.tsv'

C_OLD  = '#888888'   # grey  — old 99th-pctile
C_COMP = '#2a78d6'   # blue  — completeness
C_CONT = '#eb6834'   # orange — contamination
VARS   = [1, 10, 100]


def load():
    comp = pd.read_csv(RES_DIR / 'completeness_cutoffs.tsv',  sep='\t')
    cont = pd.read_csv(RES_DIR / 'contamination_cutoffs.tsv', sep='\t')
    old  = pd.read_csv(OLD_TSV, sep='\t')
    return comp, cont, old


# ---------------------------------------------------------------------------
# Figure 1 — Cutoff value distributions
# ---------------------------------------------------------------------------

def plot_cutoff_distributions(comp, cont, old):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        c_vals = comp[comp.variation == V]['cutoff'].dropna().values
        t_vals = cont[cont.variation == V]['cutoff'].dropna().values
        o_vals = old['cutoff_99th'].dropna().values

        lo = min(o_vals.min(), c_vals.min(), t_vals.min())
        hi = max(o_vals.max(), c_vals.max(), t_vals.max())
        bins = np.linspace(lo, hi, 60)

        ax.hist(o_vals, bins=bins, density=True, color=C_OLD,  alpha=0.5, label='Old 99th-pctile (pooled)')
        ax.hist(c_vals, bins=bins, density=True, color=C_COMP, alpha=0.6, label='Completeness (new)')
        ax.hist(t_vals, bins=bins, density=True, color=C_CONT, alpha=0.6, label='Contamination (new)')

        ymax = ax.get_ylim()[1]
        for vals, c in [(o_vals, C_OLD), (c_vals, C_COMP), (t_vals, C_CONT)]:
            med = np.median(vals)
            ax.axvline(med, color=c, linewidth=1.6, linestyle='--', alpha=0.9)
            ax.text(med, ymax * 0.97, f'{med:.1f}', color=c, fontsize=7,
                    fontweight='bold', ha='center', va='top')

        ax.set_title(f'variation = {V}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Cutoff value', fontsize=10)
        ax.set_ylabel('Density' if V == 1 else '', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    handles = [
        mpatches.Patch(facecolor=C_OLD,  alpha=0.6, edgecolor='none', label=f'Old 99th-pctile (n={len(old):,})'),
        mpatches.Patch(facecolor=C_COMP, alpha=0.6, edgecolor='none', label=f'Completeness (n={len(comp[comp.variation==1].dropna(subset=["cutoff"])):,}/variation)'),
        mpatches.Patch(facecolor=C_CONT, alpha=0.6, edgecolor='none', label=f'Contamination (n={len(cont[cont.variation==1].dropna(subset=["cutoff"])):,}/variation)'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.03),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Per-Gene Cutoff Distributions: Old 99th-Pctile vs New Specificity-Optimised',
                 fontsize=12, fontweight='bold', y=1.08)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        out = RES_DIR / f'cutoff_distributions.{ext}'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Achieved specificity distributions
# ---------------------------------------------------------------------------

def plot_specificity_distributions(comp, cont):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        c_spec = comp[comp.variation == V]['specificity'].dropna().values
        t_spec = cont[cont.variation == V]['specificity'].dropna().values

        bins = np.linspace(0, 1, 51)

        ax.hist(t_spec, bins=bins, density=True, color=C_CONT, alpha=0.6, label='Contamination')
        ax.hist(c_spec, bins=bins, density=True, color=C_COMP, alpha=0.6, label='Completeness')

        ymax = ax.get_ylim()[1]
        for vals, c in [(c_spec, C_COMP), (t_spec, C_CONT)]:
            med = np.median(vals)
            ax.axvline(med, color=c, linewidth=1.6, linestyle='--', alpha=0.9)
            ax.text(med, ymax * 0.97, f'{med:.2f}', color=c, fontsize=7.5,
                    fontweight='bold', ha='center', va='top')

        ax.set_title(f'variation = {V}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Achieved specificity', fontsize=10)
        ax.set_ylabel('Density' if V == 1 else '', fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    handles = [
        mpatches.Patch(facecolor=C_COMP, alpha=0.6, edgecolor='none', label='Completeness genes'),
        mpatches.Patch(facecolor=C_CONT, alpha=0.6, edgecolor='none', label='Contamination genes'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.03),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle('Achieved Specificity per Gene (at 99.9% Sensitivity Requirement)',
                 fontsize=12, fontweight='bold', y=1.08)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        out = RES_DIR / f'specificity_distributions.{ext}'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Step usage bar chart
# ---------------------------------------------------------------------------

def plot_step_usage(comp, cont):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    step_colors = {1: '#2ecc71', 2: '#f39c12', 3: '#9b59b6'}
    step_labels = {1: 'Step 1: perfect separation',
                   2: 'Step 2: 100% specificity',
                   3: 'Step 3: sweep'}
    x = np.array([0, 1])
    width = 0.22

    for ax, V in zip(axes, VARS):
        c_sub = comp[comp.variation == V].dropna(subset=['cutoff'])
        t_sub = cont[cont.variation == V].dropna(subset=['cutoff'])

        c_total = len(c_sub)
        t_total = len(t_sub)

        bottom_c = np.zeros(1)
        bottom_t = np.zeros(1)

        for step in [1, 2, 3]:
            c_n = (c_sub['step_used'] == step).sum()
            t_n = (t_sub['step_used'] == step).sum()
            c_pct = c_n / c_total * 100
            t_pct = t_n / t_total * 100

            ax.bar([0], [c_pct], width=0.5, bottom=bottom_c,
                   color=step_colors[step], edgecolor='white', linewidth=0.5)
            ax.bar([1], [t_pct], width=0.5, bottom=bottom_t,
                   color=step_colors[step], edgecolor='white', linewidth=0.5,
                   label=step_labels[step] if V == 1 else '_')

            if c_pct > 3:
                ax.text(0, float(bottom_c) + c_pct / 2, f'{c_pct:.0f}%',
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            if t_pct > 3:
                ax.text(1, float(bottom_t) + t_pct / 2, f'{t_pct:.0f}%',
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

            bottom_c += c_pct
            bottom_t += t_pct

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Completeness\n(n={c_total:,})', f'Contamination\n(n={t_total:,})'], fontsize=9)
        ax.set_title(f'variation = {V}', fontweight='bold', fontsize=11)
        ax.set_ylabel('% of genes' if V == 1 else '', fontsize=10)
        ax.set_ylim(0, 105)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)

    handles = [mpatches.Patch(facecolor=step_colors[s], edgecolor='none', label=step_labels[s])
               for s in [1, 2, 3]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.03),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Algorithm Step Used per Gene (Completeness vs Contamination)',
                 fontsize=12, fontweight='bold', y=1.08)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        out = RES_DIR / f'step_usage.{ext}'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print('Loading data ...')
    comp, cont, old = load()
    print(f'  completeness: {len(comp):,} rows  contamination: {len(cont):,} rows  old: {len(old):,} rows')

    print('\nFigure 1 — cutoff distributions ...')
    plot_cutoff_distributions(comp, cont, old)

    print('\nFigure 2 — specificity distributions ...')
    plot_specificity_distributions(comp, cont)

    print('\nFigure 3 — step usage ...')
    plot_step_usage(comp, cont)

    print('\nDone.')


if __name__ == '__main__':
    main()
