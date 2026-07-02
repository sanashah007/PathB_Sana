#!/usr/bin/env python3
"""
Single-genome → per-gene probabilities TSV (no SHAP), reordered by genome_order_idx.

Semantics:
- genome_order_idx[i] tells which ORIGINAL row should appear at OUTPUT row i.
  We detect 0- vs 1-based and normalize internally.

Inputs:
  --input        Genome *.txt (col0: presence 0/1, col1: genome order index)
  --genes        gene_names.txt (one name per row; same order as vector)
  --pop-mean     pop_mean.txt (one float per row; aligned with gene order)
  --unlinkage    unlinkage.txt (one float per row; aligned with gene order)
  --base-module  Python module exposing `run_model(X)` (default: run_model)
  --predict-func Function name inside the module (default: run_model)
  --out          Output TSV path (default: predictions_<input_basename>.tsv)

Output TSV columns (AFTER reordering):
  gene_idx, gene_name, input_presence, genome_order_idx, pop_mean, unlinkage,
  prob, logit_prob, odds, diff_from_pop_mean, ratio_to_pop_mean
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, List, Tuple
import argparse
import importlib
import numpy as np

# ------------------------------ loaders ------------------------------ #

def load_gene_names(gene_file: Path) -> List[str]:
    if not gene_file.is_file():
        raise FileNotFoundError(f"Gene file not found: {gene_file.resolve()}")
    return [ln.rstrip("\n") for ln in gene_file.open()]

def load_input_cols(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load presence vector (col0) and genome_order_idx (col1) from the input file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    arr = np.loadtxt(path, usecols=[0, 1], dtype=np.float64)
    if arr.ndim == 1:  # single-row fallback
        arr = arr[None, :]
    v = arr[:, 0].astype(np.float32, copy=False)
    genome_idx_raw = arr[:, 1].astype(np.int64, copy=False)  # keep raw (0- or 1-based)
    return v, genome_idx_raw

def load_1col_floats(path: Path, G: int, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path.resolve()}")
    x = np.loadtxt(path, usecols=[0], dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim != 1 or x.size != G:
        raise ValueError(f"{label} length {x.size} != gene count {G}. Ensure alignment with gene_names.txt.")
    return x

# ------------------------------ utils ------------------------------ #

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    raise TypeError("Model output must be a numpy array or a torch tensor.")

def resolve_module_fn(
    module_name: str,
    func_name: str = "run_model",
    num_not_shared: int = 100,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a predictor function that calls:
        fn(X, num_not_shared=num_not_shared)
    """
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)

    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]

        # Preferred call signature (your updated model code)
        try:
            out = fn(X, num_not_shared=num_not_shared)
        except TypeError:
            # Backward compatibility fallback if someone points at an older module
            out = fn(X)

        out = _to_numpy(out)
        return out

    return _predict

def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def normalize_pointer(ptr_raw: np.ndarray, G: int) -> np.ndarray:
    """
    Convert genome_order_idx to 0-based and validate it's a permutation.
    ptr_raw[i] = original row that should appear at output row i.
    """
    pmin, pmax = int(np.min(ptr_raw)), int(np.max(ptr_raw))
    if pmin == 0 and pmax == G - 1:
        ptr0 = ptr_raw.astype(np.int64, copy=False)
    elif pmin == 1 and pmax == G:
        ptr0 = (ptr_raw - 1).astype(np.int64, copy=False)
    else:
        raise ValueError(
            f"genome_order_idx values must be 0..{G-1} or 1..{G}; got min={pmin}, max={pmax}"
        )
    # Validate permutation
    if np.any(np.sort(ptr0) != np.arange(G)):
        raise ValueError("genome_order_idx must be a permutation of 0..G-1 (no dups/missing).")
    return ptr0

# ------------------------------ writer ------------------------------ #

def write_predictions_tsv(
    out_path: Path,
    gene_names: List[str],
    genome_order_idx_raw: np.ndarray,  # NOT reordered (aligned to output row positions)
    v_out: np.ndarray,
    pop_mean_out: np.ndarray,
    unlinkage_out: np.ndarray,
    probs_out: np.ndarray,
) -> None:
    G = probs_out.size
    logits = logit(probs_out)
    odds = probs_out / (1.0 - probs_out)
    diff = probs_out - pop_mean_out
    ratio = np.divide(
        probs_out, pop_mean_out,
        out=np.full_like(probs_out, np.nan, dtype=np.float32),
        where=pop_mean_out > 0.0
    )

    with out_path.open("w") as fh:
        fh.write("\t".join([
            "gene_idx",           # 0..G-1 AFTER reordering
            "gene_name",
            "input_presence",
            "genome_order_idx",   # as in file (raw; 0- or 1-based preserved)
            "pop_mean",
            "unlinkage",
            "prob",
            "logit_prob",
            "odds",
            "diff_from_pop_mean",
            "ratio_to_pop_mean",
        ]) + "\n")
        for i in range(G):
            fh.write("\t".join([
                str(i),
                gene_names[i],
                str(int(v_out[i])),
                str(int(genome_order_idx_raw[i])),
                f"{pop_mean_out[i]:.6f}",
                f"{unlinkage_out[i]:.6f}",
                f"{probs_out[i]:.6f}",
                f"{logits[i]:.6f}",
                f"{odds[i]:.6f}",
                f"{diff[i]:.6f}",
                f"{ratio[i]:.6f}" if np.isfinite(ratio[i]) else "NaN",
            ]) + "\n")

# ------------------------------ main ------------------------------ #

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run model on a single genome vector and write per-gene probabilities with metadata, reordered by genome_order_idx.")
    p.add_argument("--input", required=True, help="Genome *.txt (col0: presence 0/1, col1: genome order index)")
    p.add_argument("--genes", default="gene_names.txt", help="gene_names.txt (one name per row; same order as vector)")
    p.add_argument("--pop-mean", default="pop_mean.txt", help="pop_mean.txt aligned with gene order")
    p.add_argument("--unlinkage", default="unlinkage.txt", help="unlinkage.txt aligned with gene order")
    p.add_argument("--base-module", default="run_model", help="Python module for predictions (default: run_model)")
    p.add_argument("--predict-func", default="run_model", help="Function name within the module (default: run_model)")
    p.add_argument("--predict-variation",required=True, help="Amount of genes expected to change")
    p.add_argument("--out", default=None, help="Output TSV path (default: predictions_<input_basename>.tsv)")
    args = p.parse_args(argv)
    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else Path(f"predictions_{in_path.name}.tsv")

    # Load inputs (original order = model/gene order)
    v, genome_idx_raw = load_input_cols(in_path)
    G = v.size
    gene_names = load_gene_names(Path(args.genes))
    if len(gene_names) != G:
        raise RuntimeError(f"gene_names length {len(gene_names)} != vector size {G} (file: {in_path.name})")
    pop_mean = load_1col_floats(Path(args.pop_mean), G, "pop_mean")
    unlinkage = load_1col_floats(Path(args.unlinkage), G, "unlinkage")

    # Normalize pointer and validate permutation
    order0 = normalize_pointer(genome_idx_raw, G)   # 0-based, shape [G]

    # Predict on ORIGINAL order
    predict = resolve_module_fn(args.base_module, args.predict_func,args.predict_variation)
    probs = predict(v[None, :]).reshape(-1)
    if probs.size != G:
        raise ValueError(f"Model output width {probs.size} != input genes {G}")

    # Reorder ONLY for reporting: new[i] = old[order0[i]]
    gene_names_out = [gene_names[j] for j in order0]
    v_out         = v[order0]
    pop_mean_out  = pop_mean[order0]
    unlinkage_out = unlinkage[order0]
    probs_out     = probs[order0]

    # IMPORTANT: keep genome_order_idx column aligned to OUTPUT positions i
    genome_idx_for_output = genome_idx_raw  # do NOT reorder

    # Write TSV
    write_predictions_tsv(
        out_path=out_path,
        gene_names=gene_names_out,
        genome_order_idx_raw=genome_idx_for_output,
        v_out=v_out,
        pop_mean_out=pop_mean_out,
        unlinkage_out=unlinkage_out,
        probs_out=probs_out,
    )
    print(f"Wrote predictions TSV: {out_path.resolve()}  (genes={G})")

if __name__ == "__main__":
    main()

