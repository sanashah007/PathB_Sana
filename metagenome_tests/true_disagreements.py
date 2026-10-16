#!/usr/bin/env python3
"""
true_disagreements.py
======================
Counts per-genome "true disagreements" using per-gene, variation-specific
cutoffs derived from the population perturbation experiment.

A gene g disagrees in genome i at variation V when:
    logit(prob[i,g,V]) × sign[i,g] < cutoff_99th[g, V]
    where sign = 2×presence − 1  (+1 present, −1 absent)

This matches the TP-detection criterion in population_perturbation.py.
Cutoffs come from per_gene_dor_by_variation.tsv (99th-pctile of training error
scores for each gene at each variation level). For genes with no valid cutoff
(absent from that file), the universal BASELINES[V] is used as fallback.

Memory: chunked memmap for 51,702 references — login-node safe.

Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/true_disagreements.py

Outputs -> metagenome_tests/true_disagreements_results/
    per_genome_counts.tsv
    true_disagreements_boxwhisker.png/.pdf
    true_disagreements_histogram.png/.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_HERE       = Path(__file__).resolve().parent
ROOT        = _HERE.parent
REF_CACHE   = _HERE / 'adding_disagreements_tami' / 'reference_baseline_cache'
MAG_CACHE   = _HERE / 'metagenome_natural_outliers' / 'cache'
VECTORS_DIR = _HERE / 'metagenome_vectors'
CUTOFFS_TSV = ROOT / 'population_perturbation' / 'population_perturbation_results' / 'per_gene_dor_by_variation.tsv'
OUT_DIR     = _HERE / 'true_disagreements_results'

C_MAG      = '#2a78d6'
C_REF      = '#eb6834'
G          = 5598
VARS       = [1, 10, 100]
CHUNK_SIZE = 1000
BASELINES  = {V: float(np.log((1 - V / G) / (V / G))) for V in VARS}


# ---------------------------------------------------------------------------
# Load per-gene, per-variation cutoffs into a (G, 3) array
# ---------------------------------------------------------------------------

def load_cutoffs():
    """Returns cutoffs[V] = (G,) float64 array; NaN where no cutoff available."""
    df = pd.read_csv(CUTOFFS_TSV, sep='\t', usecols=['variation', 'gene_idx', 'cutoff_99th'])
    cutoffs = {}
    for V in VARS:
        vec = np.full(G, np.nan, dtype=np.float64)
        sub = df[df['variation'] == V]
        vec[sub['gene_idx'].values] = sub['cutoff_99th'].values
        cutoffs[V] = vec
        n_valid = np.sum(~np.isnan(vec))
        print(f'  v={V}: {n_valid}/{G} genes have per-gene cutoff '
              f'({G - n_valid} fall back to baseline={BASELINES[V]:.3f})')
    return cutoffs


# ---------------------------------------------------------------------------
# Core disagreement counter
# ---------------------------------------------------------------------------

def _count_true_disagreements(logits, presence, cutoff_vec, baseline):
    """
    (N,G) logits, (N,G) presence, (G,) cutoffs → (N,) int counts.
    Returns (n_disagree, n_gene_cutoff_used, n_fallback_used).
    """
    sign  = 2.0 * presence - 1.0
    score = logits * sign          # (N, G)

    has_cutoff = ~np.isnan(cutoff_vec)             # (G,) bool
    n_gene = int(has_cutoff.sum())
    n_fall = G - n_gene

    threshold = np.where(has_cutoff, cutoff_vec, baseline)   # (G,)
    disagree  = score < threshold[np.newaxis, :]              # (N, G)

    return (
        disagree.sum(axis=1).astype(np.int16),
        n_gene,
        n_fall,
    )


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute_counts(cutoffs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(_HERE / 'assignments.tsv', sep='\t')
    mag_stems   = assignments['output_file'].str.replace(r'\.txt$', '', regex=True).tolist()
    mag_files   = assignments['output_file'].tolist()
    N_mag       = len(mag_stems)

    ref_stems = (REF_CACHE / 'baseline_stems.txt').read_text().splitlines()
    N_ref     = len(ref_stems)
    print(f'\n{N_ref:,} references  |  {N_mag} MAGs')

    print('\nLoading MAG presence vectors ...')
    mag_presence = {
        stem: np.loadtxt(VECTORS_DIR / fname, usecols=[0], dtype=np.float32)
        for stem, fname in zip(mag_stems, mag_files)
    }

    pres_mm = np.lib.format.open_memmap(str(REF_CACHE / 'baseline_presence.npy'), mode='r')

    rows = []

    for V in VARS:
        bl        = BASELINES[V]
        cutoff_v  = cutoffs[V]
        print(f'\n[v={V}]')

        # MAGs
        for stem in mag_stems:
            logits   = np.load(MAG_CACHE / f'{stem}_v{V}.npy')[np.newaxis, :]
            presence = mag_presence[stem][np.newaxis, :]
            n_dis, n_gc, n_fb = _count_true_disagreements(
                logits, presence, cutoff_v, bl)
            rows.append(dict(
                genome_id             = stem,
                genome_type           = 'MAG',
                variation             = V,
                n_true_disagreements  = int(n_dis[0]),
                n_using_gene_cutoff   = n_gc,
                n_using_fallback_baseline = n_fb,
            ))

        # References (chunked)
        print('  References ...', flush=True)
        logits_mm = np.lib.format.open_memmap(
            str(REF_CACHE / f'baseline_logits_v{V}.npy'), mode='r')

        for start in range(0, N_ref, CHUNK_SIZE):
            sl       = slice(start, start + CHUNK_SIZE)
            logits_c = logits_mm[sl].copy()
            pres_c   = pres_mm[sl].copy()
            n_dis, n_gc, n_fb = _count_true_disagreements(
                logits_c, pres_c, cutoff_v, bl)
            chunk_stems = ref_stems[start: start + CHUNK_SIZE]
            for i, s in enumerate(chunk_stems):
                rows.append(dict(
                    genome_id             = s,
                    genome_type           = 'reference',
                    variation             = V,
                    n_true_disagreements  = int(n_dis[i]),
                    n_using_gene_cutoff   = n_gc,
                    n_using_fallback_baseline = n_fb,
                ))

    pg = pd.DataFrame(rows)
    pg.to_csv(OUT_DIR / 'per_genome_counts.tsv', sep='\t', index=False)
    print(f'\n  Saved: {OUT_DIR}/per_genome_counts.tsv  ({len(rows):,} rows)')
    return pg


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(pg, cutoffs):
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    for V in VARS:
        sub  = pg[pg['variation'] == V]
        mags = sub[sub['genome_type'] == 'MAG']['n_true_disagreements'].values
        refs = sub[sub['genome_type'] == 'reference']['n_true_disagreements'].values
        gc   = sub['n_using_gene_cutoff'].iloc[0]
        fb   = sub['n_using_fallback_baseline'].iloc[0]

        pooled_std = np.sqrt(
            (mags.std(ddof=1)**2 + refs.std(ddof=1)**2) / 2)
        cohens_d = abs(mags.mean() - refs.mean()) / pooled_std if pooled_std > 0 else np.nan

        print(f'\n--- variation = {V} ---')
        print(f'  Genes using per-gene cutoff : {gc}  |  fallback baseline: {fb}')
        print(f'  MAG  (n={len(mags):3d})  '
              f'median={np.median(mags):.0f}  mean={mags.mean():.1f}  std={mags.std():.1f}')
        print(f'  Ref  (n={len(refs):,})  '
              f'median={np.median(refs):.0f}  mean={refs.mean():.1f}  std={refs.std():.1f}')
        print(f"  Cohen's d (MAG vs Ref): {cohens_d:.3f}")

    # Top 10 most-flagged genes in MAGs at v=1
    print('\n--- Top 10 most frequently flagged genes in MAGs (v=1) ---')
    mag_sub = pg[(pg['variation'] == 1) & (pg['genome_type'] == 'MAG')]
    mag_stems_list = mag_sub['genome_id'].tolist()
    N_mag = len(mag_stems_list)

    # Recompute per-gene flag counts for MAGs at v=1
    cutoff_v1  = cutoffs[1]
    bl_v1      = BASELINES[1]
    threshold  = np.where(~np.isnan(cutoff_v1), cutoff_v1, bl_v1)

    mag_presence_mat = {}
    assignments = pd.read_csv(_HERE / 'assignments.tsv', sep='\t')
    stem_to_file = dict(zip(
        assignments['output_file'].str.replace(r'\.txt$', '', regex=True),
        assignments['output_file']
    ))
    flag_counts = np.zeros(G, dtype=np.int32)
    for stem in mag_stems_list:
        logits   = np.load(MAG_CACHE / f'{stem}_v1.npy')
        presence = np.loadtxt(VECTORS_DIR / stem_to_file[stem], usecols=[0], dtype=np.float32)
        sign     = 2.0 * presence - 1.0
        score    = logits * sign
        flag_counts += (score < threshold).astype(np.int32)

    gene_names_path = ROOT / 'gene_names.txt'
    gene_names = gene_names_path.read_text().splitlines() if gene_names_path.exists() else [str(i) for i in range(G)]
    top_idx = np.argsort(flag_counts)[::-1][:10]
    for rank, idx in enumerate(top_idx, 1):
        gname = gene_names[idx].split(' ', 1)[1] if ' ' in gene_names[idx] else gene_names[idx]
        pct   = flag_counts[idx] / N_mag * 100
        print(f'  {rank:2d}. gene {idx:4d}  flagged in {flag_counts[idx]:3d}/{N_mag} MAGs ({pct:.0f}%)  {gname[:60]}')


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _legend(fig):
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=C_MAG, edgecolor='none', label='MAG assemblies'),
            mpatches.Patch(facecolor=C_REF, edgecolor='none', label='Training-set references'),
        ],
        loc='upper center', bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, fontsize=10,
    )


def make_boxwhisker(pg):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        sub  = pg[pg['variation'] == V]
        mags = sub[sub['genome_type'] == 'MAG']['n_true_disagreements'].values
        refs = sub[sub['genome_type'] == 'reference']['n_true_disagreements'].values

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
        ax.set_title(f'variation = {V}', fontweight='bold', fontsize=11)
        ax.set_ylabel('True disagreements per genome' if V == 1 else '', fontsize=10)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=9)

        for pos, vals, c in [(0, mags, C_MAG), (1, refs, C_REF)]:
            ax.text(pos + 0.22, np.median(vals),
                    f'med={np.median(vals):.0f}', fontsize=7.5,
                    color=c, fontweight='bold', va='center')

    _legend(fig)
    fig.suptitle(
        'PathB True Disagreements (per-gene cutoffs): MAGs vs References',
        fontsize=12, fontweight='bold', y=1.06)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        out = OUT_DIR / f'true_disagreements_boxwhisker.{ext}'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {out}')
    plt.close(fig)


def make_histogram(pg):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        sub  = pg[pg['variation'] == V]
        mags = sub[sub['genome_type'] == 'MAG']['n_true_disagreements'].values.astype(float)
        refs = sub[sub['genome_type'] == 'reference']['n_true_disagreements'].values.astype(float)

        lo  = int(min(mags.min(), refs.min()))
        hi  = int(max(mags.max(), refs.max()))
        bins = np.linspace(lo, hi, 60)

        ax.hist(refs, bins=bins, density=True, color=C_REF, alpha=0.55, label='Reference')
        ax.hist(mags, bins=bins, density=True, color=C_MAG, alpha=0.75, label='MAG')

        ymax = ax.get_ylim()[1]
        for vals, c in [(refs, C_REF), (mags, C_MAG)]:
            med = np.median(vals)
            ax.axvline(med, color=c, linewidth=1.8, linestyle='--', alpha=0.85)
            ax.text(med + (hi - lo) * 0.01, ymax * 0.97,
                    f'med={med:.0f}', color=c, fontsize=7.5,
                    fontweight='bold', va='top')

        ax.set_title(f'variation = {V}', fontweight='bold', fontsize=11)
        ax.set_xlabel('True disagreements per genome', fontsize=10)
        ax.set_ylabel('Density' if V == 1 else '', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    _legend(fig)
    fig.suptitle(
        'PathB True Disagreements (per-gene cutoffs) — Distribution',
        fontsize=12, fontweight='bold', y=1.06)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        out = OUT_DIR / f'true_disagreements_histogram.{ext}'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print('Loading per-gene cutoffs ...')
    cutoffs = load_cutoffs()

    pg = compute_counts(cutoffs)

    print('\nGenerating plots ...')
    make_boxwhisker(pg)
    make_histogram(pg)

    print_summary(pg, cutoffs)
    print('\nDone.')


if __name__ == '__main__':
    main()
