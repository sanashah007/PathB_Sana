#!/usr/bin/env python3
"""
apply_youden_cutoff.py
=======================
Applies the pooled Youden-optimal cutoff (signed logit <= 2.30) to MAG and
train-set reference genomes, then generates comparison plots.

NOTE: val genomes are not in the baseline cache (which was built from the
51,702 train genomes only). Train references are used as the comparison group.

A gene is flagged when:
    signed_logit = logit(prob) x ((presence x 2) - 1)
    flag = signed_logit <= 2.30

MAG logits:  metagenome_natural_outliers/cache/{stem}_v{V}.npy  (5598,) float32
Ref logits:  reference_baseline_cache/baseline_logits_v{V}.npy  (51702, 5598)
             all 51,702 train references used

Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/adding_disagreements_tami/apply_youden_cutoff.py

Outputs -> metagenome_tests/adding_disagreements_tami/youden_cutoff_results/
    per_genome_flagged.tsv
    per_gene_flag_rates.tsv
    summary.txt
    youden_histogram.png/.pdf
    youden_violin.png/.pdf
    youden_ecdf.png/.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

_HERE       = Path(__file__).resolve().parent
ROOT        = _HERE.parent.parent
REF_CACHE   = _HERE / 'reference_baseline_cache'
CACHE_DIR   = _HERE.parent / 'metagenome_natural_outliers' / 'cache'
VECTORS_DIR = _HERE.parent / 'metagenome_vectors'
DOR_TSV     = ROOT / 'metagenome_tests' / 'metagenome_perturbation_results' / 'per_gene_dor.tsv'
RESULTS_DIR = _HERE / 'youden_cutoff_results'

G      = 5598
VARS   = [1, 10, 100]
CUTOFF = 2.30

C_MAG = '#2a78d6'
C_REF = '#eb6834'


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def _style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── MAG metadata ──────────────────────────────────────────────────────────
    assignments = pd.read_csv(_HERE.parent / 'assignments.tsv', sep='\t')
    mag_stems   = assignments['output_file'].str.replace(r'\.txt$', '', regex=True).tolist()
    mag_files   = assignments['output_file'].tolist()
    N_mag       = len(mag_stems)

    # ── Train references (all in baseline cache) ──────────────────────────────
    # Val genomes are not in the baseline cache; train refs are used instead.
    ref_stems = (REF_CACHE / 'baseline_stems.txt').read_text().splitlines()
    N_ref     = len(ref_stems)
    print(f'Train references in cache: {N_ref:,}')
    print(f'MAGs: {N_mag}')

    # ── MAG presence vectors ──────────────────────────────────────────────────
    print('Loading MAG presence vectors ...')
    mag_presence = {
        stem: np.loadtxt(VECTORS_DIR / fname, usecols=[0], dtype=np.float32)
        for stem, fname in zip(mag_stems, mag_files)
    }

    # ── Per-gene DOR lookup ───────────────────────────────────────────────────
    dor_df = pd.read_csv(DOR_TSV, sep='\t')
    gene_lookup = {
        int(row['gene_idx']): (row['gene_name'], row['category'], float(row['log10_dor']))
        for _, row in dor_df.iterrows()
        if int(row['gene_idx']) < G
    }

    # ── Reference memmap (shared across variations) ───────────────────────────
    pres_mm = np.lib.format.open_memmap(str(REF_CACHE / 'baseline_presence.npy'), mode='r')

    genome_rows    = []
    gene_flag_rows = []

    for V in VARS:
        print(f'\n[v={V}]')

        logits_mm = np.lib.format.open_memmap(str(REF_CACHE / f'baseline_logits_v{V}.npy'), mode='r')

        # Train references — load all rows in chunks to keep memory bounded
        print('  Train references ...', flush=True)
        CHUNK = 1000
        ref_n_flagged         = np.zeros(N_ref, dtype=np.int32)
        ref_n_absent_flagged  = np.zeros(N_ref, dtype=np.int32)
        ref_n_present_flagged = np.zeros(N_ref, dtype=np.int32)
        ref_gene_flag_count   = np.zeros(G, dtype=np.int64)

        for start in range(0, N_ref, CHUNK):
            sl              = slice(start, start + CHUNK)
            ref_logits      = logits_mm[sl].copy()
            ref_presence    = pres_mm[sl].copy()
            ref_sign        = (ref_presence * 2.0) - 1.0
            ref_flag        = (ref_logits * ref_sign) <= CUTOFF
            ref_n_flagged[sl]         = ref_flag.sum(axis=1)
            ref_n_absent_flagged[sl]  = (ref_flag & (ref_presence == 0)).sum(axis=1)
            ref_n_present_flagged[sl] = (ref_flag & (ref_presence == 1)).sum(axis=1)
            ref_gene_flag_count      += ref_flag.sum(axis=0)

        ref_gene_flag_rate = ref_gene_flag_count / N_ref

        for i, stem in enumerate(ref_stems):
            genome_rows.append(dict(
                genome_id         = stem,
                genome_type       = 'reference',
                variation         = V,
                n_flagged         = int(ref_n_flagged[i]),
                flagged_rate      = float(ref_n_flagged[i]) / G,
                n_absent_flagged  = int(ref_n_absent_flagged[i]),
                n_present_flagged = int(ref_n_present_flagged[i]),
            ))

        # MAGs
        print('  MAGs ...', flush=True)
        mag_flag_matrix = np.zeros((N_mag, G), dtype=bool)

        for i, stem in enumerate(mag_stems):
            logits   = np.load(CACHE_DIR / f'{stem}_v{V}.npy')   # (G,)
            presence = mag_presence[stem]
            sign     = (presence * 2.0) - 1.0
            flag     = (logits * sign) <= CUTOFF
            mag_flag_matrix[i] = flag
            genome_rows.append(dict(
                genome_id         = stem,
                genome_type       = 'MAG',
                variation         = V,
                n_flagged         = int(flag.sum()),
                flagged_rate      = float(flag.sum()) / G,
                n_absent_flagged  = int((flag & (presence == 0)).sum()),
                n_present_flagged = int((flag & (presence == 1)).sum()),
            ))

        mag_gene_flag_rate = mag_flag_matrix.mean(axis=0)  # (G,)

        for g, (gname, cat, ldor) in gene_lookup.items():
            gene_flag_rows.append(dict(
                gene_name       = gname,
                variation       = V,
                category        = cat,
                log10_dor       = ldor,
                flag_rate_mag   = float(mag_gene_flag_rate[g]),
                flag_rate_ref   = float(ref_gene_flag_rate[g]),
                flag_rate_delta = float(mag_gene_flag_rate[g] - ref_gene_flag_rate[g]),
            ))

        # Console stats
        mag_vals = mag_flag_matrix.sum(axis=1).astype(float)
        ref_vals = ref_n_flagged.astype(float)
        d        = cohens_d(mag_vals, ref_vals)
        ref_med  = np.median(ref_vals)
        frac_above = float((mag_vals > ref_med).mean())
        print(f'  Median n_flagged  MAG: {np.median(mag_vals):.0f}   Ref: {ref_med:.0f}')
        print(f"  Cohen's d: {d:.3f}")
        print(f'  Fraction MAG > ref median: {frac_above:.3f}')

    # ── Save TSVs ─────────────────────────────────────────────────────────────
    pg = pd.DataFrame(genome_rows)
    gf = pd.DataFrame(gene_flag_rows)

    per_genome_path = RESULTS_DIR / 'per_genome_flagged.tsv'
    per_gene_path   = RESULTS_DIR / 'per_gene_flag_rates.tsv'
    pg.to_csv(per_genome_path, sep='\t', index=False, float_format='%.6f')
    gf.to_csv(per_gene_path,   sep='\t', index=False, float_format='%.6f')
    print(f'\nSaved: {per_genome_path}  ({len(pg):,} rows)')
    print(f'Saved: {per_gene_path}  ({len(gf):,} rows)')

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = [
        f'Youden cutoff analysis  (signed logit <= {CUTOFF})',
        '=' * 55,
        f'Train references: {N_ref:,}   MAGs: {N_mag}',
        '',
    ]
    for V in VARS:
        sub      = pg[pg['variation'] == V]
        mag_vals = sub[sub['genome_type'] == 'MAG']['n_flagged'].values.astype(float)
        ref_vals = sub[sub['genome_type'] == 'reference']['n_flagged'].values.astype(float)
        d        = cohens_d(mag_vals, ref_vals)
        ref_med  = np.median(ref_vals)
        frac     = float((mag_vals > ref_med).mean())
        lines += [
            f'--- variation = {V} ---',
            f'  Train ref (n={len(ref_vals):,})  median={np.median(ref_vals):.0f}  mean={np.mean(ref_vals):.1f}',
            f'  MAG     (n={len(mag_vals):,})   median={np.median(mag_vals):.0f}  mean={np.mean(mag_vals):.1f}',
            f"  Cohen's d: {d:.3f}",
            f'  Fraction MAG > ref median: {frac:.3f}',
            '',
        ]
    lines += [
        'Previous raw disagreement counts (logit*sign < 0, all 51702 refs):',
        '  v=1:   MAG median=0   Ref median=0',
        '  v=10:  MAG median=12  Ref median=6',
        '  v=100: MAG median=89  Ref median=45',
    ]
    summary_path = RESULTS_DIR / 'summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n')
    print(f'Saved: {summary_path}')
    print('\n' + '\n'.join(lines))

    # ── Top 10 most flagged genes in MAGs at v=100 ────────────────────────────
    gf100 = gf[gf['variation'] == 100].sort_values('flag_rate_mag', ascending=False)
    print('\nTop 10 most frequently flagged genes in MAGs at v=100:')
    print(f'  {"gene_name":<60} {"category":<12} {"log10_dor":>9}  {"rate_mag":>8}  {"rate_ref":>8}')
    for _, r in gf100.head(10).iterrows():
        print(f'  {r["gene_name"]:<60} {r["category"]:<12} {r["log10_dor"]:>9.3f}  {r["flag_rate_mag"]:>8.3f}  {r["flag_rate_ref"]:>8.3f}')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\nRendering figures ...')
    _plot_histograms(pg, N_mag, N_ref)
    _plot_violins(pg, N_mag, N_ref)
    _plot_ecdf(pg, N_mag, N_ref)
    print('\nDone.')


# ── Figure 1 — Overlapping density histograms ─────────────────────────────────

def _plot_histograms(pg, N_mag, N_ref):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        sub  = pg[pg['variation'] == V]
        mags = sub[sub['genome_type'] == 'MAG']['n_flagged'].values.astype(float)
        refs = sub[sub['genome_type'] == 'reference']['n_flagged'].values.astype(float)

        lo   = 0
        hi   = max(mags.max(), refs.max()) + 5
        bins = np.linspace(lo, hi, 60)

        ax.hist(refs, bins=bins, density=True, alpha=0.6, color=C_REF,
                label=f'Train references (n={N_ref:,})')
        ax.hist(mags, bins=bins, density=True, alpha=0.6, color=C_MAG,
                label=f'MAGs (n={N_mag})')

        ymax = ax.get_ylim()[1]
        for vals, color in [(mags, C_MAG), (refs, C_REF)]:
            med = float(np.median(vals))
            ax.axvline(med, color=color, linestyle='--', linewidth=1.5)
            ax.text(med + hi * 0.01, ymax * 0.94, f'med={med:.0f}',
                    color=color, fontsize=8, va='top')

        ax.set_title(f'variation = {V}', fontsize=11, fontweight='bold')
        ax.set_xlabel('n_flagged per genome', fontsize=10)
        ax.set_ylabel('Density' if V == 1 else '', fontsize=10)
        _style_ax(ax)
        if V == 1:
            ax.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    _save_fig(fig, 'youden_histogram')


# ── Figure 2 — Violin with strip ──────────────────────────────────────────────

def _plot_violins(pg, N_mag, N_ref):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    order   = ['MAG', 'reference']
    palette = {'MAG': C_MAG, 'reference': C_REF}

    for ax, V in zip(axes, VARS):
        sub  = pg[pg['variation'] == V].copy()
        mags = sub[sub['genome_type'] == 'MAG']
        refs = sub[sub['genome_type'] == 'reference']
        combo = pd.concat([mags[['genome_type', 'n_flagged']],
                           refs[['genome_type', 'n_flagged']]])

        sns.violinplot(data=combo, x='genome_type', y='n_flagged', hue='genome_type',
                       order=order, hue_order=order, palette=palette,
                       ax=ax, inner=None, linewidth=1.2, cut=0, legend=False)

        # Individual MAG points
        sns.stripplot(data=mags, x='genome_type', y='n_flagged', hue='genome_type',
                      order=order, hue_order=order, palette=palette,
                      ax=ax, size=3.5, jitter=True, alpha=0.65, zorder=4, legend=False)

        # Mean markers
        for pos, grp in [(0, mags), (1, refs)]:
            ax.scatter([pos], [grp['n_flagged'].mean()],
                       marker='D', s=40, color='white',
                       edgecolors='#333333', linewidths=1.2, zorder=5)

        ax.set_title(f'variation = {V}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('n_flagged per genome' if V == 1 else '', fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'MAG\n(n={N_mag})', f'Train refs\n(n={N_ref:,})'], fontsize=9)
        _style_ax(ax)

    handles = [
        mpatches.Patch(facecolor=C_MAG, edgecolor='none', label=f'MAGs (n={N_mag})'),
        mpatches.Patch(facecolor=C_REF, edgecolor='none', label=f'Train references (n={N_ref:,})'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor='#333333', markersize=7, label='Mean'),
    ]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, 'youden_violin')


# ── Figure 3 — Empirical CDF ──────────────────────────────────────────────────

def _plot_ecdf(pg, N_mag, N_ref):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('white')

    for ax, V in zip(axes, VARS):
        sub  = pg[pg['variation'] == V]
        mags = np.sort(sub[sub['genome_type'] == 'MAG']['n_flagged'].values.astype(float))
        refs = np.sort(sub[sub['genome_type'] == 'reference']['n_flagged'].values.astype(float))

        mag_y = np.arange(1, len(mags) + 1) / len(mags)
        ref_y = np.arange(1, len(refs) + 1) / len(refs)

        ax.step(refs, ref_y, color=C_REF, linewidth=2.0,
                label=f'Train refs (n={N_ref:,})', where='post')
        ax.step(mags, mag_y, color=C_MAG, linewidth=2.0,
                label=f'MAGs (n={N_mag})', where='post')

        # Shade between curves on common x grid
        x_all   = np.sort(np.unique(np.concatenate([mags, refs])))
        mag_cdf = np.searchsorted(mags, x_all, side='right') / len(mags)
        ref_cdf = np.searchsorted(refs, x_all, side='right') / len(refs)
        ax.fill_between(x_all, ref_cdf, mag_cdf, alpha=0.12, color='#888888')

        # Max separation
        sep     = np.abs(mag_cdf - ref_cdf)
        idx_max = int(np.argmax(sep))
        x_sep   = float(x_all[idx_max])
        ax.axvline(x_sep, color='#555555', linestyle=':', linewidth=1.3)
        ax.text(x_sep * 1.01 + 0.5, 0.50,
                f'Δmax={sep[idx_max]:.2f}\nx={x_sep:.0f}',
                fontsize=8, color='#555555', va='center')

        ax.set_title(f'variation = {V}', fontsize=11, fontweight='bold')
        ax.set_xlabel('n_flagged per genome', fontsize=10)
        ax.set_ylabel('Fraction of genomes ≤ x' if V == 1 else '', fontsize=10)
        ax.set_ylim(0, 1.05)
        _style_ax(ax)
        if V == 1:
            ax.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    _save_fig(fig, 'youden_ecdf')


def _save_fig(fig, name):
    for suffix in ('png', 'pdf'):
        out = RESULTS_DIR / f'{name}.{suffix}'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
