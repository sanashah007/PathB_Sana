#!/usr/bin/env python3
"""
completeness_contamination_scores.py
=====================================
Computes per-genome completeness and contamination scores from PathB logit
outputs without re-running the model.

Concept
-------
Baseline logit for ~100 gene changes out of 5598:
    baseline = logit(1 - 100/5598) ~= +4.007

Completeness score (sum over ABSENT genes, input=0):
    sum(logit[g] + 4)
    Negative -> model agrees genes are absent   (complete assembly)
    Positive -> model expects missing genes     (incomplete assembly)

Contamination score (sum over PRESENT genes, input=1):
    sum(logit[g] - 4)
    Positive -> model agrees genes are present  (clean genome)
    Negative -> model doubts present genes      (potential contamination)

Memory: chunked memmap for references (~250 MB peak) -- safe on login node.
Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/completeness_contamination_scores.py

Outputs -> metagenome_tests/completeness_contamination_results/
    per_genome_scores.tsv
    per_gene_contributions.tsv
    summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path

_HERE       = Path(__file__).resolve().parent
ROOT        = _HERE.parent
REF_CACHE   = _HERE / 'adding_disagreements_tami' / 'reference_baseline_cache'
MAG_CACHE   = _HERE / 'metagenome_natural_outliers' / 'cache'
VECTORS_DIR = _HERE / 'metagenome_vectors'
OUT_DIR     = _HERE / 'completeness_contamination_results'

G          = 5598
VARS       = [1, 10, 100]
CHUNK_SIZE = 1000
BASELINES  = {V: float(np.log((1 - V / G) / (V / G))) for V in VARS}


def _scores(logits, presence, baseline):
    """Per-genome completeness and contamination scores. (N,G) -> (N,),(N,),(N,),(N,)"""
    absent  = presence < 0.5
    present = ~absent
    comp = logits + baseline
    cont = logits - baseline
    return (
        np.where(absent,  comp, 0.0).sum(axis=1),
        np.where(present, cont, 0.0).sum(axis=1),
        absent.sum(axis=1),
        present.sum(axis=1),
    )


def _gene_contributions(logits, presence, baseline):
    """Partial sums for per-gene contributions. (N,G) -> 4x (G,) accumulators.
    Accumulate across chunks, then call _finalize_contributions."""
    absent  = presence < 0.5
    present = ~absent
    comp = logits.astype(np.float64) + baseline
    cont = logits.astype(np.float64) - baseline
    return (
        np.where(absent,  comp, 0.0).sum(axis=0),
        absent.sum(axis=0).astype(np.int32),
        np.where(present, cont, 0.0).sum(axis=0),
        present.sum(axis=0).astype(np.int32),
    )


def _finalize_contributions(comp_sum, comp_cnt, cont_sum, cont_cnt):
    """Convert accumulated sums/counts to per-gene means (NaN where count=0)."""
    safe_cc = np.where(comp_cnt > 0, comp_cnt, 1)
    safe_tc = np.where(cont_cnt > 0, cont_cnt, 1)
    comp_mean = np.where(comp_cnt > 0, comp_sum / safe_cc, np.nan)
    cont_mean = np.where(cont_cnt > 0, cont_sum / safe_tc, np.nan)
    return comp_mean, cont_mean


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gene_names = (ROOT / 'gene_names.txt').read_text().splitlines()
    assert len(gene_names) == G, f'Expected {G} gene names, got {len(gene_names)}'

    dor = pd.read_csv(
        ROOT / 'population_perturbation' / 'population_perturbation_results' / 'per_gene_dor.tsv',
        sep='\t', usecols=['gene_idx', 'category', 'log10_dor']
    ).set_index('gene_idx')

    assignments = pd.read_csv(_HERE / 'assignments.tsv', sep='\t')
    mag_stems   = assignments['output_file'].str.replace(r'\.txt$', '', regex=True).tolist()
    mag_files   = assignments['output_file'].tolist()
    N_mag       = len(mag_stems)

    ref_stems = (REF_CACHE / 'baseline_stems.txt').read_text().splitlines()
    N_ref     = len(ref_stems)

    for V in VARS:
        print(f'  v={V}: baseline logit = {BASELINES[V]:.6f}')
    print(f'{N_ref:,} reference genomes  |  {N_mag} MAG genomes  |  {len(VARS)} variation levels')

    print('\nLoading MAG presence vectors ...')
    mag_presence_mat = np.stack([
        np.loadtxt(VECTORS_DIR / fname, usecols=[0], dtype=np.float32)
        for fname in mag_files
    ])  # (N_mag, G)

    genome_rows = []
    gene_rows   = []

    pres_mm = np.lib.format.open_memmap(str(REF_CACHE / 'baseline_presence.npy'), mode='r')

    for V in VARS:
        print(f'\n[v={V}]')

        # MAG scores (all in memory, fast)
        print('  MAG scores ...')
        mag_logits_mat = np.stack([
            np.load(MAG_CACHE / f'{stem}_v{V}.npy') for stem in mag_stems
        ])  # (N_mag, G)

        comp_s, cont_s, n_abs, n_prs = _scores(mag_logits_mat, mag_presence_mat, BASELINES[V])
        cs, cc, ts, tc = _gene_contributions(mag_logits_mat, mag_presence_mat, BASELINES[V])
        comp_mean_mag, cont_mean_mag = _finalize_contributions(cs, cc, ts, tc)

        for i, stem in enumerate(mag_stems):
            genome_rows.append(dict(
                genome_id                      = stem,
                genome_type                    = 'MAG',
                variation                      = V,
                completeness_score             = float(comp_s[i]),
                contamination_score            = float(cont_s[i]),
                n_absent_genes                 = int(n_abs[i]),
                n_present_genes                = int(n_prs[i]),
                completeness_score_normalized  = float(comp_s[i] / n_abs[i]) if n_abs[i] > 0 else np.nan,
                contamination_score_normalized = float(cont_s[i] / n_prs[i]) if n_prs[i] > 0 else np.nan,
            ))

        # Reference scores (chunked memmap)
        print('  Reference scores (chunked) ...')
        logits_mm = np.lib.format.open_memmap(
            str(REF_CACHE / f'baseline_logits_v{V}.npy'), mode='r')

        ref_comp_s = np.zeros(N_ref, dtype=np.float32)
        ref_cont_s = np.zeros(N_ref, dtype=np.float32)
        ref_n_abs  = np.zeros(N_ref, dtype=np.int16)
        ref_n_prs  = np.zeros(N_ref, dtype=np.int16)

        comp_sum_ref = np.zeros(G, dtype=np.float64)
        comp_cnt_ref = np.zeros(G, dtype=np.int32)
        cont_sum_ref = np.zeros(G, dtype=np.float64)
        cont_cnt_ref = np.zeros(G, dtype=np.int32)

        for start in range(0, N_ref, CHUNK_SIZE):
            sl       = slice(start, start + CHUNK_SIZE)
            logits_c = logits_mm[sl].copy()
            pres_c   = pres_mm[sl].copy()

            cs, ts, na, np_ = _scores(logits_c, pres_c, BASELINES[V])
            ref_comp_s[sl] = cs
            ref_cont_s[sl] = ts
            ref_n_abs[sl]  = na
            ref_n_prs[sl]  = np_

            gc_s, gc_c, gt_s, gt_c = _gene_contributions(logits_c, pres_c, BASELINES[V])
            comp_sum_ref += gc_s
            comp_cnt_ref += gc_c
            cont_sum_ref += gt_s
            cont_cnt_ref += gt_c

        comp_mean_ref, cont_mean_ref = _finalize_contributions(
            comp_sum_ref, comp_cnt_ref, cont_sum_ref, cont_cnt_ref)

        for i, stem in enumerate(ref_stems):
            na, np_ = int(ref_n_abs[i]), int(ref_n_prs[i])
            genome_rows.append(dict(
                genome_id                      = stem,
                genome_type                    = 'reference',
                variation                      = V,
                completeness_score             = float(ref_comp_s[i]),
                contamination_score            = float(ref_cont_s[i]),
                n_absent_genes                 = na,
                n_present_genes                = np_,
                completeness_score_normalized  = float(ref_comp_s[i] / na) if na > 0 else np.nan,
                contamination_score_normalized = float(ref_cont_s[i] / np_) if np_ > 0 else np.nan,
            ))

        for g in range(G):
            row = dict(
                gene_idx                            = g,
                gene_name                           = gene_names[g],
                category    = dor.loc[g, 'category']  if g in dor.index else 'unknown',
                log10_dor   = float(dor.loc[g, 'log10_dor']) if g in dor.index else np.nan,
                variation                           = V,
                mean_completeness_contribution_mag  = float(comp_mean_mag[g]),
                mean_completeness_contribution_ref  = float(comp_mean_ref[g]),
                mean_contamination_contribution_mag = float(cont_mean_mag[g]),
                mean_contamination_contribution_ref = float(cont_mean_ref[g]),
                completeness_contribution_delta     = float(comp_mean_mag[g] - comp_mean_ref[g]),
                contamination_contribution_delta    = float(cont_mean_mag[g] - cont_mean_ref[g]),
            )
            gene_rows.append(row)

    print('\nWriting outputs ...')
    pg = pd.DataFrame(genome_rows)
    gd = pd.DataFrame(gene_rows)

    pg.to_csv(OUT_DIR / 'per_genome_scores.tsv',      sep='\t', index=False, float_format='%.6f')
    gd.to_csv(OUT_DIR / 'per_gene_contributions.tsv', sep='\t', index=False, float_format='%.6f')

    # Summary text
    pct = [25, 50, 75, 90, 99]
    lines = ['baselines: ' + '  '.join(f'v{V}={BASELINES[V]:.4f}' for V in VARS), '']

    for V in VARS:
        sub  = pg[pg['variation'] == V]
        mags = sub[sub['genome_type'] == 'MAG']
        refs = sub[sub['genome_type'] == 'reference']
        gv   = gd[gd['variation'] == V]

        def pct_str(series):
            vals = np.nanpercentile(series, pct)
            return '  '.join(f'p{p}={v:.2f}' for p, v in zip(pct, vals))

        lines += [
            f'=== variation = {V} ===',
            '',
            '--- Completeness score ---',
            f'  MAG (n={len(mags)})   mean={mags["completeness_score"].mean():.2f}  median={mags["completeness_score"].median():.2f}',
            f'  Ref (n={len(refs):,}) mean={refs["completeness_score"].mean():.2f}  median={refs["completeness_score"].median():.2f}',
            f'  MAG percentiles: {pct_str(mags["completeness_score"])}',
            f'  Ref percentiles: {pct_str(refs["completeness_score"])}',
            '',
            '--- Contamination score ---',
            f'  MAG (n={len(mags)})   mean={mags["contamination_score"].mean():.2f}  median={mags["contamination_score"].median():.2f}',
            f'  Ref (n={len(refs):,}) mean={refs["contamination_score"].mean():.2f}  median={refs["contamination_score"].median():.2f}',
            f'  MAG percentiles: {pct_str(mags["contamination_score"])}',
            f'  Ref percentiles: {pct_str(refs["contamination_score"])}',
            '',
            '--- Top 20 genes driving completeness in MAGs (most positive mean contribution) ---',
            '  (positive = model expects gene present despite it being absent in assembly)',
        ]
        top_comp = gv.nlargest(20, 'mean_completeness_contribution_mag')
        for _, r in top_comp.iterrows():
            gname = r['gene_name'].split(' ', 1)[1] if ' ' in r['gene_name'] else r['gene_name']
            lines.append(
                f'  [{r["category"]:>10}  logDOR={r["log10_dor"]:5.2f}]  '
                f'MAG={r["mean_completeness_contribution_mag"]:+.3f}  '
                f'Ref={r["mean_completeness_contribution_ref"]:+.3f}  '
                f'delta={r["completeness_contribution_delta"]:+.3f}  {gname[:60]}'
            )

        lines += [
            '',
            '--- Top 20 genes driving contamination in MAGs (most negative mean contribution) ---',
            '  (negative = model doubts gene despite it being present in assembly)',
        ]
        top_cont = gv.nsmallest(20, 'mean_contamination_contribution_mag')
        for _, r in top_cont.iterrows():
            gname = r['gene_name'].split(' ', 1)[1] if ' ' in r['gene_name'] else r['gene_name']
            lines.append(
                f'  [{r["category"]:>10}  logDOR={r["log10_dor"]:5.2f}]  '
                f'MAG={r["mean_contamination_contribution_mag"]:+.3f}  '
                f'Ref={r["mean_contamination_contribution_ref"]:+.3f}  '
                f'delta={r["contamination_contribution_delta"]:+.3f}  {gname[:60]}'
            )
        lines.append('')

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines))

    print(f'  {OUT_DIR}/per_genome_scores.tsv       ({len(genome_rows):,} rows)')
    print(f'  {OUT_DIR}/per_gene_contributions.tsv  ({len(gene_rows):,} rows)')
    print(f'  {OUT_DIR}/summary.txt')
    print('\nDone.')


if __name__ == '__main__':
    main()
