#!/usr/bin/env python3
"""
cache_reference_baselines.py
============================
Caches PathB logit outputs for a chunk of training-set reference genomes.

Processes genomes [chunk_idx * chunk_size : (chunk_idx+1) * chunk_size]
from the sorted training-set stem list and saves partial arrays to:
    <out-dir>/chunks/
        stems_{chunk_idx:04d}.txt
        presence_{chunk_idx:04d}.npy      -- (N_chunk, 5598) float32
        logits_{chunk_idx:04d}_v1.npy     -- (N_chunk, 5598) float32
        logits_{chunk_idx:04d}_v10.npy    -- (N_chunk, 5598) float32
        logits_{chunk_idx:04d}_v100.npy   -- (N_chunk, 5598) float32

Designed for SLURM array job (pass SLURM_ARRAY_TASK_ID as --chunk-idx).
Fully idempotent: exits immediately if all five outputs already exist.

CRITICAL: only training-set genomes (split == 'train') are processed.
Val/test genomes are never touched.

Must be run from PATHB_sample root where run_model.py and twolayer.pth reside.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import run_model as _rm

G = 5598
VARIATIONS = [1, 10, 100]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def load_presence(path: Path) -> np.ndarray:
    return np.loadtxt(path, usecols=[0], dtype=np.float32)


def run_model_batch(X: np.ndarray, num_not_shared: int) -> np.ndarray:
    result = _rm.run_model(X, num_not_shared=num_not_shared)
    if hasattr(result, 'detach'):
        result = result.detach().cpu().numpy()
    return _logit(np.asarray(result, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Cache PathB outputs for a chunk of training-set reference genomes.'
    )
    parser.add_argument('--chunk-idx', type=int, required=True,
                        help='0-based chunk index (use $SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Genomes per chunk (default: 1000)')
    parser.add_argument('--train-split',
                        default='population_perturbation/population_perturbation_results/train_val_test_split.tsv',
                        help='Path to train_val_test_split.tsv')
    parser.add_argument('--ref-dir', default='all_embeded_genomes/',
                        help='Directory containing reference genome .txt files')
    parser.add_argument('--out-dir', default=str(Path(__file__).resolve().parent / 'reference_baseline_cache'),
                        help='Output root directory')
    args = parser.parse_args()

    chunk_idx  = args.chunk_idx
    chunk_size = args.chunk_size
    out_dir    = Path(args.out_dir)
    chunks_dir = out_dir / 'chunks'
    chunks_dir.mkdir(parents=True, exist_ok=True)

    stems_out  = chunks_dir / f'stems_{chunk_idx:04d}.txt'
    pres_out   = chunks_dir / f'presence_{chunk_idx:04d}.npy'
    logit_outs = {V: chunks_dir / f'logits_{chunk_idx:04d}_v{V}.npy' for V in VARIATIONS}

    # Idempotency: skip if all outputs already exist
    all_outputs = [stems_out, pres_out] + list(logit_outs.values())
    if all(f.exists() for f in all_outputs):
        print(f'Chunk {chunk_idx}: all outputs already exist, skipping.')
        return

    # Load training stems (sorted for deterministic chunk assignment)
    split_df    = pd.read_csv(args.train_split, sep='\t')
    train_stems = sorted(
        split_df.loc[split_df['split'] == 'train', 'filename']
        .str.replace(r'\.txt$', '', regex=True)
        .tolist()
    )
    print(f'Training-set genomes total: {len(train_stems)}')

    start = chunk_idx * chunk_size
    end   = min(start + chunk_size, len(train_stems))
    if start >= len(train_stems):
        print(f'Chunk {chunk_idx}: start index {start} >= total {len(train_stems)}, nothing to do.')
        return

    chunk_stems = train_stems[start:end]
    n           = len(chunk_stems)
    print(f'Chunk {chunk_idx}: processing genomes [{start}:{end}] (n={n})')

    # Load presence vectors
    ref_dir = Path(args.ref_dir)
    X       = np.zeros((n, G), dtype=np.float32)
    missing = []
    for i, stem in enumerate(chunk_stems):
        genome_path = ref_dir / f'{stem}.txt'
        if not genome_path.exists():
            missing.append(stem)
        else:
            pres = load_presence(genome_path)
            if pres.shape[0] != G:
                raise ValueError(f'{genome_path}: expected {G} genes, got {pres.shape[0]}')
            X[i] = pres

    if missing:
        print(f'WARNING: {len(missing)} genomes missing from {ref_dir}. '
              f'First 3: {missing[:3]}')
        missing_out = chunks_dir / f'missing_{chunk_idx:04d}.txt'
        missing_out.write_text('\n'.join(missing) + '\n')

    # Run model at each variation level
    logit_arrays: dict[int, np.ndarray] = {}
    for V in VARIATIONS:
        print(f'  variation={V}: running on {n} genomes ...')
        logit_arrays[V] = run_model_batch(X, num_not_shared=V)

    # Write outputs
    stems_out.write_text('\n'.join(chunk_stems) + '\n')
    np.save(pres_out, X)
    for V in VARIATIONS:
        np.save(logit_outs[V], logit_arrays[V])

    print(f'Chunk {chunk_idx}: done ({n} genomes saved).')


if __name__ == '__main__':
    main()
