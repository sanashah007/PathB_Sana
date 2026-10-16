#!/usr/bin/env python3
"""
merge_reference_baselines.py
============================
Merges chunk outputs from cache_reference_baselines.py into final arrays.

Reads all completed chunks from <cache-dir>/chunks/ and writes:
    <cache-dir>/baseline_stems.txt          -- one stem per line, in chunk order
    <cache-dir>/baseline_presence.npy       -- (N, 5598) float32
    <cache-dir>/baseline_logits_v1.npy      -- (N, 5598) float32
    <cache-dir>/baseline_logits_v10.npy     -- (N, 5598) float32
    <cache-dir>/baseline_logits_v100.npy    -- (N, 5598) float32

Uses np.lib.format.open_memmap so each array is written incrementally —
RAM usage is bounded to one chunk at a time (~22 MB), regardless of total size.

Run after all SLURM array tasks complete successfully.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

G          = 5598
VARIATIONS = [1, 10, 100]
EXPECTED_N = 51702


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Merge reference baseline chunks into final aggregated arrays.'
    )
    parser.add_argument('--cache-dir', default=str(Path(__file__).resolve().parent / 'reference_baseline_cache'),
                        help='Root directory used by cache_reference_baselines.py')
    parser.add_argument('--expected-n', type=int, default=EXPECTED_N,
                        help='Expected total training-set genome count (default: 51702)')
    parser.add_argument('--delete-chunks', action='store_true',
                        help='Remove chunks/ directory after a successful merge')
    args = parser.parse_args()

    cache_dir  = Path(args.cache_dir)
    chunks_dir = cache_dir / 'chunks'

    if not chunks_dir.exists():
        sys.exit(f'ERROR: chunks directory not found: {chunks_dir}')

    # Discover completed chunks (all 5 output files must exist)
    stems_files = sorted(chunks_dir.glob('stems_*.txt'))
    if not stems_files:
        sys.exit('ERROR: no stems_*.txt files found in chunks/')

    completed   = []
    incomplete  = []
    for sf in stems_files:
        idx = int(sf.stem.split('_')[1])
        pf  = chunks_dir / f'presence_{idx:04d}.npy'
        lf  = {V: chunks_dir / f'logits_{idx:04d}_v{V}.npy' for V in VARIATIONS}
        if pf.exists() and all(lf[V].exists() for V in VARIATIONS):
            completed.append(idx)
        else:
            incomplete.append(idx)

    if incomplete:
        print(f'WARNING: {len(incomplete)} incomplete chunks will be skipped: {incomplete}')

    completed.sort()
    if not completed:
        sys.exit('ERROR: no complete chunks found.')

    print(f'Complete chunks: {len(completed)} '
          f'(indices {completed[0]}..{completed[-1]})')

    # Count total genomes across completed chunks
    chunk_sizes: dict[int, int] = {}
    for idx in completed:
        sf     = chunks_dir / f'stems_{idx:04d}.txt'
        stems  = [s.strip() for s in sf.read_text().splitlines() if s.strip()]
        chunk_sizes[idx] = len(stems)

    n_total = sum(chunk_sizes.values())
    print(f'Total genomes: {n_total} (expected: {args.expected_n})')
    if n_total != args.expected_n:
        print(f'WARNING: count mismatch — check for missing/extra chunks.')

    # Create output arrays via memmapped .npy files (writes proper header + data)
    pres_path   = cache_dir / 'baseline_presence.npy'
    logit_paths = {V: cache_dir / f'baseline_logits_v{V}.npy' for V in VARIATIONS}

    print('Allocating output arrays ...')
    pres_out   = np.lib.format.open_memmap(
        pres_path, mode='w+', dtype='float32', shape=(n_total, G)
    )
    logit_outs = {
        V: np.lib.format.open_memmap(
            logit_paths[V], mode='w+', dtype='float32', shape=(n_total, G)
        )
        for V in VARIATIONS
    }

    all_stems: list[str] = []
    offset = 0
    for idx in completed:
        n  = chunk_sizes[idx]
        sf = chunks_dir / f'stems_{idx:04d}.txt'
        chunk_stems = [s.strip() for s in sf.read_text().splitlines() if s.strip()]
        all_stems.extend(chunk_stems)

        pres_chunk = np.load(chunks_dir / f'presence_{idx:04d}.npy')
        pres_out[offset:offset + n] = pres_chunk
        del pres_chunk

        for V in VARIATIONS:
            logit_chunk = np.load(chunks_dir / f'logits_{idx:04d}_v{V}.npy')
            logit_outs[V][offset:offset + n] = logit_chunk
            del logit_chunk

        offset += n
        if idx % 10 == 0 or idx == completed[-1]:
            print(f'  chunk {idx:04d} merged  [{offset}/{n_total}]')

    # Flush memory-mapped arrays
    del pres_out
    for V in VARIATIONS:
        del logit_outs[V]

    # Write stems index
    stems_path = cache_dir / 'baseline_stems.txt'
    stems_path.write_text('\n'.join(all_stems) + '\n')

    print('\nMerge complete:')
    print(f'  {stems_path}: {len(all_stems)} stems')
    print(f'  {pres_path}: ({n_total}, {G}) float32')
    for V in VARIATIONS:
        size_gb = (n_total * G * 4) / 1e9
        print(f'  {logit_paths[V]}: ({n_total}, {G}) float32  [{size_gb:.2f} GB]')

    if args.delete_chunks:
        shutil.rmtree(chunks_dir)
        print(f'\nDeleted: {chunks_dir}')


if __name__ == '__main__':
    main()
