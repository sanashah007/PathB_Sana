#!/usr/bin/env python3
"""
compute_disagreement_metrics.py
================================
Counts raw PathB model-input disagreements per genome.

A gene g in genome i disagrees at variation V when:
    logit[i,g,V] × sign[i,g] < 0
    where sign = 2×presence − 1  (+1 present, −1 absent)

i.e. the model's output directly contradicts the input gene presence,
regardless of magnitude.

Memory: chunked memmap for references (~200 MB peak) — safe on login node.
Run from PATHB_sample root:
    conda run -n PathB python metagenome_tests/adding_disagreements_tami/compute_disagreement_metrics.py

Outputs (in disagreement_distribution_results/):
    per_genome_summary.tsv
    distribution_summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

_HERE       = Path(__file__).resolve().parent
REF_CACHE   = _HERE / 'reference_baseline_cache'
RESULTS_DIR = _HERE / 'disagreement_distribution_results'
CACHE_DIR   = _HERE.parent / 'metagenome_natural_outliers' / 'cache'
VECTORS_DIR = _HERE.parent / 'metagenome_vectors'

G          = 5598
VARS       = [1, 10, 100]
CHUNK_SIZE = 1000


def _disagree(logits, presence):
    """Count disagreements per genome. (N,G) -> (N,) int16."""
    sign = 2.0 * presence - 1.0
    return (logits * sign < 0).sum(axis=1).astype(np.int16)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(_HERE.parent / 'assignments.tsv', sep='\t')
    mag_stems   = assignments['output_file'].str.replace(r'\.txt$', '', regex=True).tolist()
    mag_files   = assignments['output_file'].tolist()
    N_mag       = len(mag_stems)

    ref_stems = (REF_CACHE / 'baseline_stems.txt').read_text().splitlines()
    N_ref     = len(ref_stems)

    print(f'{N_ref:,} references  |  {N_mag} MAGs  |  {len(VARS)} variations')

    print('\nLoading MAG presence vectors ...')
    mag_presence = {
        stem: np.loadtxt(VECTORS_DIR / fname, usecols=[0], dtype=np.float32)
        for stem, fname in zip(mag_stems, mag_files)
    }

    pres_mm = np.lib.format.open_memmap(str(REF_CACHE / 'baseline_presence.npy'), mode='r')

    genome_rows = []

    for V in VARS:
        print(f'\n[v={V}]')

        # References (chunked)
        print('  References ...', flush=True)
        logits_mm   = np.lib.format.open_memmap(str(REF_CACHE / f'baseline_logits_v{V}.npy'), mode='r')
        ref_n_dis   = np.zeros(N_ref, dtype=np.int16)
        for start in range(0, N_ref, CHUNK_SIZE):
            sl = slice(start, start + CHUNK_SIZE)
            ref_n_dis[sl] = _disagree(logits_mm[sl].copy(), pres_mm[sl].copy())

        for i, stem in enumerate(ref_stems):
            genome_rows.append(dict(
                genome_id         = stem,
                genome_type       = 'reference',
                variation         = V,
                n_genes           = G,
                n_disagreements   = int(ref_n_dis[i]),
                disagreement_rate = float(ref_n_dis[i]) / G,
            ))

        # MAGs
        print('  MAGs ...', flush=True)
        for i, stem in enumerate(mag_stems):
            cache_path = CACHE_DIR / f'{stem}_v{V}.npy'
            if not cache_path.exists():
                raise FileNotFoundError(f'MAG logit cache missing: {cache_path}')
            logits   = np.load(cache_path)[np.newaxis, :]
            presence = mag_presence[stem][np.newaxis, :]
            n_dis    = int(_disagree(logits, presence)[0])
            genome_rows.append(dict(
                genome_id         = stem,
                genome_type       = 'MAG',
                variation         = V,
                n_genes           = G,
                n_disagreements   = n_dis,
                disagreement_rate = n_dis / G,
            ))

    pg = pd.DataFrame(genome_rows)
    pg.to_csv(RESULTS_DIR / 'per_genome_summary.tsv', sep='\t', index=False)

    lines = ['Raw disagreements (logit × sign < 0)', '']
    for V in VARS:
        sub  = pg[pg['variation'] == V]
        mags = sub[sub['genome_type'] == 'MAG']
        refs = sub[sub['genome_type'] == 'reference']
        lines += [
            f'--- variation = {V} ---',
            f'  MAG (n={len(mags):,})  median={mags["n_disagreements"].median():.0f}  '
            f'mean={mags["n_disagreements"].mean():.1f}  '
            f'rate={mags["disagreement_rate"].mean():.4f}',
            f'  Ref (n={len(refs):,})  median={refs["n_disagreements"].median():.0f}  '
            f'mean={refs["n_disagreements"].mean():.1f}  '
            f'rate={refs["disagreement_rate"].mean():.4f}',
            '',
        ]
    (RESULTS_DIR / 'distribution_summary.txt').write_text('\n'.join(lines))

    print(f'\n  {RESULTS_DIR}/per_genome_summary.tsv  ({len(genome_rows):,} rows)')
    print(f'  {RESULTS_DIR}/distribution_summary.txt')
    print('\nDone.')


if __name__ == '__main__':
    main()
