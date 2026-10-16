#!/usr/bin/env python3
"""
plot_disagreements_baseline.py
================================
Box-and-whisker plots of baseline-adjusted PathB disagreement counts.

A gene disagrees at variation V when:
    logit × sign < BASELINE[V]
    where sign = 2×presence − 1  (+1 present, −1 absent)
    and BASELINE[V] = logit(1 − V/5598)  (same baseline as completeness/contamination)

    v=1  : BASELINE ≈ 8.63
    v=10 : BASELINE ≈ 6.33
    v=100: BASELINE ≈ 4.01

This is stricter than raw (< 0): it flags genes where the model is less confident
than expected, not just genes where the model crosses zero.

Reads directly from cached logit arrays — no separate compute step needed.
Memory: chunked memmap for references (~200 MB peak) — safe on login node.

Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/adding_disagreements_tami/plot_disagreements_baseline.py

Outputs:
    metagenome_tests/adding_disagreements_tami/disagreement_plots_baseline.png
    metagenome_tests/adding_disagreements_tami/disagreement_histogram_baseline.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_HERE       = Path(__file__).resolve().parent
REF_CACHE   = _HERE / 'reference_baseline_cache'
MAG_CACHE   = _HERE.parent / 'metagenome_natural_outliers' / 'cache'
VECTORS_DIR = _HERE.parent / 'metagenome_vectors'

C_MAG = '#2a78d6'
C_REF = '#eb6834'
G     = 5598
VARS  = [1, 10, 100]
CHUNK_SIZE = 1000

BASELINES = {V: float(np.log((1 - V / G) / (V / G))) for V in VARS}


def _count_disagreements(logits, presence, baseline):
    """(N,G) arrays → (N,) int16 count of genes where logit*sign < baseline."""
    sign = 2.0 * presence - 1.0
    return (logits * sign < baseline).sum(axis=1).astype(np.int16)


def compute_counts():
    assignments = pd.read_csv(_HERE.parent / 'assignments.tsv', sep='\t')
    mag_stems   = assignments['output_file'].str.replace(r'\.txt$', '', regex=True).tolist()
    mag_files   = assignments['output_file'].tolist()
    N_mag       = len(mag_stems)

    ref_stems = (REF_CACHE / 'baseline_stems.txt').read_text().splitlines()
    N_ref     = len(ref_stems)

    print(f'Baselines: ' + '  '.join(f'v{V}={BASELINES[V]:.3f}' for V in VARS))
    print(f'{N_ref:,} references  |  {N_mag} MAGs')

    print('\nLoading MAG presence vectors ...')
    mag_presence = {
        stem: np.loadtxt(VECTORS_DIR / fname, usecols=[0], dtype=np.float32)
        for stem, fname in zip(mag_stems, mag_files)
    }

    pres_mm = np.lib.format.open_memmap(str(REF_CACHE / 'baseline_presence.npy'), mode='r')

    # results[V] = {'mag': (N_mag,), 'ref': (N_ref,)} arrays of disagreement counts
    results = {}

    for V in VARS:
        bl = BASELINES[V]
        print(f'\n[v={V}  baseline={bl:.3f}]')

        # MAGs
        mag_counts = np.array([
            int(_count_disagreements(
                np.load(MAG_CACHE / f'{stem}_v{V}.npy')[np.newaxis, :],
                mag_presence[stem][np.newaxis, :],
                bl
            )[0])
            for stem in mag_stems
        ], dtype=np.int16)

        # References (chunked)
        print('  References ...', flush=True)
        logits_mm  = np.lib.format.open_memmap(str(REF_CACHE / f'baseline_logits_v{V}.npy'), mode='r')
        ref_counts = np.zeros(N_ref, dtype=np.int16)
        for start in range(0, N_ref, CHUNK_SIZE):
            sl = slice(start, start + CHUNK_SIZE)
            ref_counts[sl] = _count_disagreements(
                logits_mm[sl].copy(), pres_mm[sl].copy(), bl)

        results[V] = {'mag': mag_counts, 'ref': ref_counts}

        print(f'  MAG  median={np.median(mag_counts):.0f}  mean={mag_counts.mean():.1f}')
        print(f'  Ref  median={np.median(ref_counts):.0f}  mean={ref_counts.mean():.1f}')

    return results


def make_plot(results):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        mags = results[V]['mag']
        refs = results[V]['ref']

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

        jitter = rng.uniform(-0.15, 0.15, len(mags))
        ax.scatter(jitter, mags, color=C_MAG, alpha=0.7, s=20, zorder=5, linewidths=0)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'MAG\n(n={len(mags)})', f'Ref\n(n={len(refs):,})'], fontsize=9)
        ax.set_title(f'variation = {V}\n(baseline = {BASELINES[V]:.2f})',
                     fontweight='bold', fontsize=11)
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
    fig.suptitle(
        'PathB Baseline-Adjusted Disagreements (logit × sign < baseline): MAGs vs References',
        fontsize=12, fontweight='bold', y=1.06)

    plt.tight_layout()
    out = _HERE / 'disagreement_plots_baseline.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {out}')
    plt.close(fig)


def make_histogram(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        mags = results[V]['mag'].astype(float)
        refs = results[V]['ref'].astype(float)

        all_vals = np.concatenate([mags, refs])
        lo, hi = int(all_vals.min()), int(all_vals.max())
        bins = np.linspace(lo, hi, 60)

        ax.hist(refs, bins=bins, density=True, color=C_REF, alpha=0.55, label='Reference')
        ax.hist(mags, bins=bins, density=True, color=C_MAG, alpha=0.75, label='MAG')

        for vals, c in [(refs, C_REF), (mags, C_MAG)]:
            med = np.median(vals)
            ax.axvline(med, color=c, linewidth=1.8, linestyle='--', alpha=0.85)
            ax.text(med + (hi - lo) * 0.01, ax.get_ylim()[1] * 0.97,
                    f'med={med:.0f}', color=c, fontsize=7.5,
                    fontweight='bold', va='top')

        ax.set_title(f'variation = {V}\n(baseline = {BASELINES[V]:.2f})',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Disagreements per genome', fontsize=10)
        ax.set_ylabel('Density' if V == 1 else '', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    legend_handles = [
        mpatches.Patch(facecolor=C_MAG, edgecolor='none', alpha=0.75, label='MAG assemblies'),
        mpatches.Patch(facecolor=C_REF, edgecolor='none', alpha=0.55, label='Training-set references'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=10)
    fig.suptitle(
        'PathB Baseline-Adjusted Disagreements — Distribution (logit × sign < baseline)',
        fontsize=12, fontweight='bold', y=1.06)

    plt.tight_layout()
    out = _HERE / 'disagreement_histogram_baseline.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
    plt.close(fig)


def main():
    results = compute_counts()
    make_plot(results)
    make_histogram(results)
    print('Done.')


if __name__ == '__main__':
    main()
