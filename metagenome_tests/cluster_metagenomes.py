#!/usr/bin/env python3
"""
cluster_metagenomes.py
======================
Featurize assembled metagenome FASTAs into the same 2-column presence/absence
format used by all_embeded_genomes/.

Pipeline per FASTA:
  1. mlst          -> sequence type (ST) for the output filename
  2. Bakta         -> annotated proteins (.faa)
  3. esm-extract   -> per-gene ESM2 embeddings (.pt files)
  4. cluster       -> assign each gene to nearest centroid in new_clusters3.pkl
                      (threshold 0.32, same as the original pipeline)
  5. save vector   -> {name}__{ST}__{rel_year}__{col_year}.txt  (2 columns)
  6. log           -> closest existing reference from --ref-vectors-dir
                      (Hamming distance on column 0)

Output format (matches cluster_with_locations.py / all_embeded_genomes/):
  5598 lines, two space-separated integers per line
  col 0: presence (1=present, 0=absent)
  col 1: order of appearance (1-based; 0 if absent)

Requirements:
  conda activate PathB
  pip install fair-esm==2.0.1        # provides esm-extract CLI
  --ref-clusters  new_clusters3.pkl  # from SuperCloud: /home/gridsan/ptorrillo/dates_included/
  --esm-model     esm2_t6_8M_UR50D.pt
  --bakta-db      /path/to/db-light  # from SuperCloud: /pool001/torrillo/bakta_db_light/db-light

Usage:
  python cluster_metagenomes.py \\
      --fasta-dir  /path/to/mag_fastas/ \\
      --ref-clusters new_clusters3.pkl \\
      --esm-model  esm2_t6_8M_UR50D.pt \\
      --bakta-db   /path/to/db-light \\
      --output-dir metagenome_vectors/ \\
      --log-file   assignments.tsv
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pickle
import torch
from scipy.spatial.distance import cdist


# =============================================================================
# STEP 1 — MLST
# =============================================================================

def run_mlst(fasta_path):
    """
    Run mlst and return the sequence type string (e.g. '8').
    mlst stdout: <filename> <scheme> <ST> <allele1> ...
    Returns '0' if mlst fails or ST is '-'.
    """
    result = subprocess.run(
        ['mlst', str(fasta_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [mlst] warning: non-zero exit for {fasta_path.name}", file=sys.stderr)
        return '0'
    parts = result.stdout.strip().split('\t')
    st = parts[2] if len(parts) > 2 else '0'
    return st if st != '-' else '0'


# =============================================================================
# STEP 2 — BAKTA ANNOTATION
# =============================================================================

def run_bakta(fasta_path, work_dir, bakta_db, threads=8):
    """
    Annotate a FASTA with Bakta.
    Returns the path to the output .faa protein file.
    """
    bakta_out = work_dir / 'bakta_out'
    bakta_out.mkdir(exist_ok=True)
    cmd = [
        'bakta', str(fasta_path),
        '--db',           bakta_db,
        '--output',       str(bakta_out),
        '--genus',        'Staphylococcus',
        '--species',      'aureus',
        '--threads',      str(threads),
        '--skip-trna', '--skip-tmrna', '--skip-rrna',
        '--skip-ncrna', '--skip-ncrna-region', '--skip-crispr',
        '--skip-pseudo', '--skip-sorf', '--skip-gap', '--skip-ori',
        '--skip-plot', '--force',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Bakta failed:\n{result.stderr[-2000:]}")

    faa_files = list(bakta_out.glob('*.faa'))
    if not faa_files:
        raise FileNotFoundError(f"Bakta produced no .faa file in {bakta_out}")
    return faa_files[0]


# =============================================================================
# STEP 3 — ESM2 EMBEDDINGS
# =============================================================================

def run_esm_extract(faa_path, work_dir, esm_model_path, toks_per_batch=4096):
    """
    Embed proteins using the ESM2 Python API (equivalent to esm-extract --include mean).
    Saves one .pt file per protein with {'mean_representations': {6: tensor}}.
    Returns the directory containing .pt files.
    """
    import esm as esm_module

    esm_out = work_dir / 'esm_out'
    esm_out.mkdir(exist_ok=True)

    model, alphabet = esm_module.pretrained.load_model_and_alphabet_local(esm_model_path)
    model.eval()

    dataset = esm_module.FastaBatchedDataset.from_file(str(faa_path))
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    loader  = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(),
        batch_sampler=batches,
    )

    with torch.no_grad():
        for batch_labels, batch_strs, batch_tokens in loader:
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
            for i, label in enumerate(batch_labels):
                seq_len   = len(batch_strs[i])
                mean_rep  = results['representations'][6][i, 1:seq_len + 1].mean(0)
                safe_label = label.replace('/', '_').replace('\\', '_')
                torch.save(
                    {'mean_representations': {6: mean_rep}},
                    esm_out / f"{safe_label}.pt",
                )

    return esm_out


# =============================================================================
# STEP 4 — CLUSTER ASSIGNMENT  (adapted from cluster_with_locations.py)
# =============================================================================

def _parse_pt_filename(filename):
    """Sort key: extract (contig_idx, start_pos) from filename if present."""
    match = re.search(r'contig_(\d+)_(\d+)_\d+\.pt', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')


def assign_clusters(esm_dir, ref_embeddings, threshold=0.32):
    """
    Load .pt files, compare each gene embedding to cluster centroids,
    and return a (5598, 2) presence/order array.

    Matches cluster_with_locations.py exactly:
      col 0: 1 if nearest centroid distance < threshold, else 0
      col 1: 1-based order of first appearance (0 if absent)
    """
    pt_files = sorted(
        Path(esm_dir).rglob('*.pt'),
        key=lambda p: _parse_pt_filename(p.name),
    )

    presence = np.zeros((len(ref_embeddings), 2), dtype=int)
    order_counter = 1

    for pt_file in pt_files:
        data = torch.load(str(pt_file))
        gene_emb = data['mean_representations'][6].numpy()
        dists = cdist([gene_emb], ref_embeddings)
        closest = int(np.argmin(dists))
        if dists[0, closest] < threshold:
            if presence[closest, 0] == 0:
                presence[closest, 0] = 1
                presence[closest, 1] = order_counter
                order_counter += 1

    return presence


# =============================================================================
# STEP 5 — FIND CLOSEST REFERENCE  (Hamming distance on col 0)
# =============================================================================

def load_ref_vectors(ref_vectors_dir):
    """
    Load all .txt files from ref_vectors_dir into a matrix.
    Returns (names_list, matrix of shape (N, 5598) bool).
    Memory: ~90k * 5598 bytes = ~500 MB — loaded once.
    """
    paths = sorted(Path(ref_vectors_dir).glob('*.txt'))
    if not paths:
        return [], None

    names = [p.stem for p in paths]
    vectors = np.stack(
        [np.loadtxt(p, usecols=[0], dtype=np.uint8) for p in paths],
        axis=0,
    )  # (N, 5598)
    return names, vectors


def find_closest_reference(presence_col0, ref_names, ref_matrix):
    """
    Return (closest_name, hamming_distance) for the nearest reference vector.
    Hamming distance = fraction of genes that differ (0=identical, 1=opposite).
    """
    if ref_matrix is None:
        return 'N/A', float('nan')
    diffs = (ref_matrix != presence_col0[None, :]).sum(axis=1)
    best = int(np.argmin(diffs))
    return ref_names[best], int(diffs[best])


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Featurize metagenome FASTAs into presence/absence vectors.'
    )
    parser.add_argument('--fasta-dir',      required=True,
                        help='Directory of input FASTA files (.fna / .fa / .fasta)')
    parser.add_argument('--ref-clusters',
                        default='/orcd/data/tami/003/projects/PATHB_sample/extra_files_for_embedding_genomes/new_clusters3.pkl',
                        help='new_clusters3.pkl — 5598 ESM2 cluster centroids')
    parser.add_argument('--esm-model',
                        default='/orcd/data/tami/003/projects/PATHB_sample/extra_files_for_embedding_genomes/esm2_t6_8M_UR50D.pt',
                        help='Path to esm2_t6_8M_UR50D.pt')
    parser.add_argument('--bakta-db',
                        default='/home/sanashah/PATHB_sample/metagenome_tests/bakta_db',
                        help='Path to Bakta database directory (db-light)')
    parser.add_argument('--ref-vectors-dir',
                        default='/orcd/data/tami/003/projects/PATHB_sample/all_embeded_genomes/',
                        help='Directory of reference .txt vectors for closest-match logging')
    parser.add_argument('--output-dir',     default='metagenome_vectors/',
                        help='Output directory for generated .txt vectors')
    parser.add_argument('--log-file',       default='assignments.tsv',
                        help='TSV log: genome, ST, n_genes_present, closest_ref, hamming')
    parser.add_argument('--threshold',      type=float, default=0.32,
                        help='Euclidean distance cutoff for cluster assignment (default 0.32)')
    parser.add_argument('--release-year',   default='0000',
                        help='Release year for output filename (default 0000)')
    parser.add_argument('--collection-year', default='0000',
                        help='Collection year for output filename (default 0000)')
    parser.add_argument('--threads',        type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Load reference cluster centroids
    # -------------------------------------------------------------------------
    print(f"Loading cluster centroids from {args.ref_clusters} ...")
    with open(args.ref_clusters, 'rb') as f:
        raw = pickle.load(f)
    ref_embeddings = np.array([coords for coords, _, _ in raw])
    print(f"  {len(ref_embeddings)} clusters loaded")

    # -------------------------------------------------------------------------
    # Load reference vectors for closest-match logging
    # -------------------------------------------------------------------------
    print(f"Loading reference vectors from {args.ref_vectors_dir} ...")
    ref_names, ref_matrix = load_ref_vectors(args.ref_vectors_dir)
    print(f"  {len(ref_names)} reference vectors loaded")

    # -------------------------------------------------------------------------
    # Discover input FASTAs
    # -------------------------------------------------------------------------
    fasta_dir = Path(args.fasta_dir)
    fastas = sorted(
        list(fasta_dir.glob('*.fna')) +
        list(fasta_dir.glob('*.fa'))  +
        list(fasta_dir.glob('*.fasta'))
    )
    print(f"\nFound {len(fastas)} FASTA files in {fasta_dir}")
    if not fastas:
        sys.exit('No FASTA files found. Check --fasta-dir.')

    # -------------------------------------------------------------------------
    # Process each FASTA
    # -------------------------------------------------------------------------
    log_rows = []

    for fasta in fastas:
        name = fasta.stem
        print(f"\n[{name}]")

        with tempfile.TemporaryDirectory(prefix=f'metagenome_{name}_') as tmp:
            tmp = Path(tmp)
            try:
                # Step 1: ST
                print('  mlst ...')
                st = run_mlst(fasta)
                print(f'  ST = {st}')

                # Step 2: Bakta
                print('  Bakta annotation ...')
                faa_path = run_bakta(fasta, tmp, args.bakta_db, args.threads)
                print(f'  proteins: {faa_path}')

                # Step 3: ESM2
                print('  esm-extract ...')
                esm_dir = run_esm_extract(faa_path, tmp, args.esm_model, args.threads)
                n_pt = len(list(esm_dir.rglob('*.pt')))
                print(f'  {n_pt} .pt files generated')

                # Step 4: cluster assignment
                print('  assigning to clusters ...')
                presence = assign_clusters(esm_dir, ref_embeddings, args.threshold)
                n_present = int(presence[:, 0].sum())
                print(f'  genes present: {n_present} / {len(ref_embeddings)}')

                # Step 5: find closest reference
                closest_ref, hamming = find_closest_reference(
                    presence[:, 0], ref_names, ref_matrix
                )
                print(f'  closest reference: {closest_ref}  (hamming={hamming})')

                # Save vector
                out_name = f"{name}__{st}__{args.release_year}__{args.collection_year}.txt"
                out_path = output_dir / out_name
                np.savetxt(out_path, presence, fmt='%d', delimiter=' ')
                print(f'  saved -> {out_path}')

                log_rows.append({
                    'genome':        name,
                    'output_file':   out_name,
                    'ST':            st,
                    'n_genes_present': n_present,
                    'closest_ref':   closest_ref,
                    'hamming':       hamming,
                })

            except Exception as e:
                print(f'  ERROR: {e}', file=sys.stderr)
                log_rows.append({
                    'genome':        name,
                    'output_file':   '',
                    'ST':            '',
                    'n_genes_present': '',
                    'closest_ref':   '',
                    'hamming':       '',
                    'error':         str(e),
                })

    # -------------------------------------------------------------------------
    # Write log
    # -------------------------------------------------------------------------
    import csv
    log_path = Path(args.log_file)
    fieldnames = ['genome', 'output_file', 'ST', 'n_genes_present',
                  'closest_ref', 'hamming', 'error']
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore',
                                delimiter='\t')
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nDone. {len(log_rows)} genomes processed.")
    print(f"Vectors saved to: {output_dir}/")
    print(f"Log saved to:     {log_path}")


if __name__ == '__main__':
    main()
