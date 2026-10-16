#!/usr/bin/env python3
"""
gene_cutoff_specificity.py
==========================
Sensitivity-first per-gene classification cutoffs, computed separately for
completeness genes (present→absent flips, input=0) and contamination genes
(absent→present flips, input=1).

Scoring convention (same as population_perturbation.py):
    score = logit(prob) × sign    where sign = 2×presence − 1
    Lower score → more error-like.
    score <= cutoff  → classified as error (positive detection).
    score >  cutoff  → classified as non-error (negative).

Algorithm (both completeness and contamination use the same steps):
    Step 1  Perfect separation — use highest error score as cutoff.
    Step 2  100% specificity achievable — use lowest nonerror score.
    Step 3  Sweep — find lowest cutoff satisfying min_sensitivity.
    Step 4  Edge case — relax to best achievable sensitivity.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-gene cutoff functions
# ---------------------------------------------------------------------------

def _cutoff_core(
    error_scores: np.ndarray,
    nonerror_scores: np.ndarray,
    min_sensitivity: float,
    score_type: str,
) -> dict:
    """
    Shared algorithm for completeness and contamination cutoffs.
    error_scores   : 1-D signed logit scores where gene was genuinely an error.
    nonerror_scores: 1-D signed logit scores where gene was genuinely correct.
    score_type     : 'completeness' or 'contamination'.
    """
    err = np.asarray(error_scores,   dtype=np.float64)
    ne  = np.asarray(nonerror_scores, dtype=np.float64)
    n_err = len(err)
    n_ne  = len(ne)

    def _metrics(cutoff: float):
        sensitivity = float((err <= cutoff).mean()) if n_err > 0 else 0.0
        specificity = float((ne  >  cutoff).mean()) if n_ne  > 0 else 1.0
        return sensitivity, specificity

    def _counts(cutoff: float):
        tp = int((err <= cutoff).sum())
        fn = int((err >  cutoff).sum())
        fp = int((ne  <= cutoff).sum())
        tn = int((ne  >  cutoff).sum())
        return tp, fn, fp, tn

    def _dor(tp, fp, fn, tn):
        denom = fp * fn
        dor   = (tp * tn) / denom if denom > 0 else float('nan')
        log10 = math.log10(dor) if (dor > 0 and not math.isnan(dor)) else float('nan')
        return dor, log10

    def _build(cutoff, step, sens_met, overlapping=True):
        tp, fn, fp, tn = _counts(cutoff)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        dor, log10_dor = _dor(tp, fp, fn, tn)
        return {
            'cutoff':                    float(cutoff),
            'sensitivity':               sensitivity,
            'specificity':               specificity,
            'n_error':                   n_err,
            'n_nonerror':                n_ne,
            'fp':                        fp,
            'fn':                        fn,
            'tp':                        tp,
            'tn':                        tn,
            'overlapping':               overlapping,
            'step_used':                 step,
            'sensitivity_requirement_met': sens_met,
            'dor':                       dor,
            'log10_dor':                 log10_dor,
            'score_type':                score_type,
        }

    if n_err == 0:
        # No error instances — return NaN result
        return {
            'cutoff': float('nan'), 'sensitivity': float('nan'),
            'specificity': float('nan'), 'n_error': 0, 'n_nonerror': n_ne,
            'fp': 0, 'fn': 0, 'tp': 0, 'tn': n_ne,
            'overlapping': False, 'step_used': 0,
            'sensitivity_requirement_met': False,
            'dor': float('nan'), 'log10_dor': float('nan'),
            'score_type': score_type,
        }

    max_err  = float(err.max())
    min_ne   = float(ne.min()) if n_ne > 0 else float('inf')

    # ------------------------------------------------------------------
    # Step 1: Perfect separation
    # ------------------------------------------------------------------
    if max_err < min_ne:
        return _build(max_err, step=1, sens_met=True, overlapping=False)

    # ------------------------------------------------------------------
    # Step 2: 100% specificity cutoff
    # ------------------------------------------------------------------
    if n_ne > 0:
        cutoff2 = min_ne - np.finfo(np.float64).eps * abs(min_ne) - np.finfo(np.float64).tiny
        # Use the value just below min_ne so all non-errors are above it.
        # Practically: set to min_ne itself; score > cutoff means non-errors
        # with score == min_ne pass (not flagged). Use min_ne as cutoff.
        cutoff2 = min_ne
        sens2, _ = _metrics(cutoff2)
        if sens2 >= min_sensitivity:
            return _build(cutoff2, step=2, sens_met=True)

    # ------------------------------------------------------------------
    # Step 3: Sweep all candidate cutoffs
    # ------------------------------------------------------------------
    # Candidates are every unique score value across both sets.
    candidates = np.unique(np.concatenate([err, ne]))

    best_cutoff    = None
    best_spec      = -1.0
    sens_met_sweep = False

    for c in candidates:
        s, sp = _metrics(float(c))
        if s < min_sensitivity:
            continue
        sens_met_sweep = True
        if sp > best_spec or (sp == best_spec and (best_cutoff is None or c < best_cutoff)):
            best_spec   = sp
            best_cutoff = float(c)

    if best_cutoff is not None:
        return _build(best_cutoff, step=3, sens_met=True)

    # ------------------------------------------------------------------
    # Step 4: No cutoff meets min_sensitivity — relax to best sensitivity
    # ------------------------------------------------------------------
    best_cutoff2 = float(candidates[-1])   # highest possible → max sensitivity
    return _build(best_cutoff2, step=4, sens_met=False)


def compute_completeness_cutoff(
    error_scores: np.ndarray,
    nonerror_scores: np.ndarray,
    min_sensitivity: float = 0.999,
) -> dict:
    """
    Per-gene cutoff optimised for completeness detection (present→absent flips).
    Requires min_sensitivity fraction of error_scores be <= cutoff,
    then maximises specificity among qualifying cutoffs.
    """
    return _cutoff_core(error_scores, nonerror_scores, min_sensitivity, 'completeness')


def compute_contamination_cutoff(
    error_scores: np.ndarray,
    nonerror_scores: np.ndarray,
    min_sensitivity: float = 0.999,
) -> dict:
    """
    Per-gene cutoff optimised for contamination detection (absent→present flips).
    Same algorithm as compute_completeness_cutoff.
    """
    return _cutoff_core(error_scores, nonerror_scores, min_sensitivity, 'contamination')


# ---------------------------------------------------------------------------
# Batch functions
# ---------------------------------------------------------------------------

def _batch_cutoffs(
    scores: np.ndarray,
    error_mask: np.ndarray,
    presence_matrix: np.ndarray,
    gene_names: list,
    min_sensitivity: float,
    min_errors: int,
    score_type: str,
) -> pd.DataFrame:
    """
    Shared implementation for completeness and contamination batch functions.

    scores          : (n_genomes, n_genes) signed logit scores
    error_mask      : (n_genomes, n_genes) bool, True where gene was flipped in
                      the relevant direction (present→absent or absent→present)
    presence_matrix : (n_genomes, n_genes) ORIGINAL (pre-flip) presence values
    score_type      : 'completeness' or 'contamination'
    """
    n_genomes, n_genes = scores.shape
    assert error_mask.shape   == (n_genomes, n_genes), "error_mask shape mismatch"
    assert presence_matrix.shape == (n_genomes, n_genes), "presence_matrix shape mismatch"

    # Non-error selection: rows that were NOT flipped AND have the relevant
    # original presence direction.
    if score_type == 'completeness':
        # completeness: errors are present→absent flips (original presence=1)
        # non-errors:   genuinely absent genes (not flipped, original presence=0)
        nonerror_selector = (~error_mask) & (presence_matrix == 0)
        n_col = 'n_completeness_errors'
    else:
        # contamination: errors are absent→present flips (original presence=0)
        # non-errors:    genuinely present genes (not flipped, original presence=1)
        nonerror_selector = (~error_mask) & (presence_matrix == 1)
        n_col = 'n_contamination_errors'

    fn = compute_completeness_cutoff if score_type == 'completeness' else compute_contamination_cutoff

    rows = []
    for g in range(n_genes):
        err_scores = scores[error_mask[:, g],      g]
        ne_scores  = scores[nonerror_selector[:, g], g]

        n_err = len(err_scores)
        result = {'gene_name': gene_names[g], n_col: n_err}

        if n_err < min_errors:
            result.update({
                'cutoff': float('nan'), 'sensitivity': float('nan'),
                'specificity': float('nan'), 'n_error': n_err,
                'n_nonerror': len(ne_scores),
                'fp': float('nan'), 'fn': float('nan'),
                'tp': float('nan'), 'tn': float('nan'),
                'overlapping': float('nan'), 'step_used': float('nan'),
                'sensitivity_requirement_met': float('nan'),
                'dor': float('nan'), 'log10_dor': float('nan'),
                'score_type': score_type,
            })
        else:
            result.update(fn(err_scores, ne_scores, min_sensitivity))

        rows.append(result)

    # Re-order columns so gene_name and count come first
    df = pd.DataFrame(rows)
    first_cols = ['gene_name', n_col]
    other_cols = [c for c in df.columns if c not in first_cols]
    return df[first_cols + other_cols].reset_index(drop=True)


def compute_all_completeness_cutoffs(
    scores: np.ndarray,
    error_mask: np.ndarray,
    presence_matrix: np.ndarray,
    gene_names: list,
    min_sensitivity: float = 0.999,
    min_errors: int = 50,
) -> pd.DataFrame:
    """
    Compute completeness cutoffs for every gene.

    scores          : (n_genomes, n_genes) — logit × sign, from CORRUPTED input
    error_mask      : (n_genomes, n_genes) — True where gene was flipped present→absent
    presence_matrix : (n_genomes, n_genes) — ORIGINAL (pre-flip) presence values
    gene_names      : list of G gene name strings
    """
    return _batch_cutoffs(
        scores, error_mask, presence_matrix,
        gene_names, min_sensitivity, min_errors, 'completeness',
    )


def compute_all_contamination_cutoffs(
    scores: np.ndarray,
    error_mask: np.ndarray,
    presence_matrix: np.ndarray,
    gene_names: list,
    min_sensitivity: float = 0.999,
    min_errors: int = 50,
) -> pd.DataFrame:
    """
    Compute contamination cutoffs for every gene.

    scores          : (n_genomes, n_genes) — logit × sign, from CORRUPTED input
    error_mask      : (n_genomes, n_genes) — True where gene was flipped absent→present
    presence_matrix : (n_genomes, n_genes) — ORIGINAL (pre-flip) presence values
    gene_names      : list of G gene name strings
    """
    return _batch_cutoffs(
        scores, error_mask, presence_matrix,
        gene_names, min_sensitivity, min_errors, 'contamination',
    )
