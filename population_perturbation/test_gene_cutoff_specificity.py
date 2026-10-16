#!/usr/bin/env python3
"""
test_gene_cutoff_specificity.py
================================
Unit tests for gene_cutoff_specificity.py.

Run from PATHB_sample root:
    conda run -n PathB python -m pytest population_perturbation/test_gene_cutoff_specificity.py -v
Or directly:
    conda run -n PathB python population_perturbation/test_gene_cutoff_specificity.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gene_cutoff_specificity import (
    compute_completeness_cutoff,
    compute_contamination_cutoff,
    compute_all_completeness_cutoffs,
    compute_all_contamination_cutoffs,
)

REQUIRED_KEYS = {
    'cutoff', 'sensitivity', 'specificity',
    'n_error', 'n_nonerror', 'fp', 'fn', 'tp', 'tn',
    'overlapping', 'step_used', 'sensitivity_requirement_met',
    'dor', 'log10_dor', 'score_type',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_keys(result):
    assert REQUIRED_KEYS == set(result.keys()), (
        f"Missing keys: {REQUIRED_KEYS - set(result.keys())}, "
        f"Extra keys: {set(result.keys()) - REQUIRED_KEYS}"
    )


def _check_counts_consistent(r):
    """TP+FN == n_error and FP+TN == n_nonerror."""
    if math.isnan(r['cutoff']):
        return
    assert r['tp'] + r['fn'] == r['n_error'],   f"tp+fn={r['tp']+r['fn']} != n_error={r['n_error']}"
    assert r['fp'] + r['tn'] == r['n_nonerror'], f"fp+tn={r['fp']+r['tn']} != n_nonerror={r['n_nonerror']}"


# ---------------------------------------------------------------------------
# Step 1: Perfect separation
# ---------------------------------------------------------------------------

class TestPerfectSeparation:
    def test_completeness_no_overlap(self):
        err = np.array([-5.0, -3.0, -1.0])
        ne  = np.array([ 2.0,  4.0,  6.0])
        r   = compute_completeness_cutoff(err, ne)
        _check_keys(r)
        _check_counts_consistent(r)
        assert r['overlapping']             is False
        assert r['step_used']               == 1
        assert r['sensitivity']             == 1.0
        assert r['specificity']             == 1.0
        assert r['sensitivity_requirement_met'] is True
        assert r['score_type']              == 'completeness'
        assert r['cutoff']                  == pytest.approx(-1.0)

    def test_contamination_no_overlap(self):
        err = np.array([-8.0, -6.0, -4.0])
        ne  = np.array([ 1.0,  3.0,  5.0])
        r   = compute_contamination_cutoff(err, ne)
        _check_keys(r)
        assert r['overlapping']  is False
        assert r['step_used']    == 1
        assert r['sensitivity']  == 1.0
        assert r['specificity']  == 1.0
        assert r['score_type']   == 'contamination'

    def test_single_error_single_nonerror_separated(self):
        r = compute_completeness_cutoff(
            error_scores=np.array([-2.0]),
            nonerror_scores=np.array([3.0]),
        )
        _check_keys(r)
        assert r['step_used'] == 1
        assert r['sensitivity'] == 1.0
        assert r['specificity'] == 1.0
        assert r['overlapping'] is False


# ---------------------------------------------------------------------------
# Step 2: 100% specificity achievable
# ---------------------------------------------------------------------------

class TestStep2SpecificityCutoff:
    def test_100pct_specificity_with_sufficient_sensitivity(self):
        # max_err=5.5 > min_ne=5.0 → overlap, Step 1 skipped.
        # Step 2: cutoff=5.0, sens=fraction([1,2,3,4,5.5]<=5.0)=4/5=0.8 >= 0.8 → fires.
        err = np.array([1.0, 2.0, 3.0, 4.0, 5.5])
        ne  = np.array([5.0, 6.0, 7.0, 8.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=0.8)
        _check_keys(r)
        _check_counts_consistent(r)
        assert r['step_used']    == 2
        assert r['sensitivity']  >= 0.8
        assert r['sensitivity_requirement_met'] is True

    def test_step2_fails_if_sensitivity_too_low(self):
        # Step 2: cutoff=5, sens=fraction([1,2,10]<=5)=2/3 < 0.999 → fail → step 3
        err = np.array([1.0, 2.0, 10.0])
        ne  = np.array([5.0, 6.0, 7.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=0.999)
        assert r['step_used'] in (3, 4)


# ---------------------------------------------------------------------------
# Step 3: Sweep needed
# ---------------------------------------------------------------------------

class TestStep3Sweep:
    def test_sweep_finds_highest_specificity(self):
        # Errors: [-2, 0, 3, 5]  non-errors: [1, 4, 6, 8, 10]
        # min_sensitivity=0.75 → need >= 3 errors captured
        # Candidates: -2, 0, 1, 3, 4, 5, 6, 8, 10
        # At cutoff=3: sens=3/4=0.75, spec=fraction(ne>3)=4/5=0.8 ✓
        # At cutoff=4: sens=3/4=0.75, spec=3/5=0.6 — not better
        # At cutoff=5: sens=4/4=1.0, spec=3/5=0.6
        # At cutoff=1: sens=2/4=0.5 < 0.75, skip
        # Best at 0.75 sens: cutoff=3, spec=0.8
        err = np.array([-2.0, 0.0, 3.0, 5.0])
        ne  = np.array([ 1.0, 4.0, 6.0, 8.0, 10.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=0.75)
        _check_keys(r)
        _check_counts_consistent(r)
        assert r['step_used']   == 3
        assert r['sensitivity'] >= 0.75
        assert r['sensitivity_requirement_met'] is True
        # Highest specificity should be at cutoff=3
        assert r['cutoff']      == pytest.approx(3.0)
        assert r['specificity'] == pytest.approx(0.8)

    def test_sweep_tiebreak_lowest_cutoff(self):
        # err=[1,2,4,5], ne=[3,3,9,9], min_sens=0.75
        # Step 1: max_err=5 > min_ne=3 → overlap.
        # Step 2: cutoff=3, sens=fraction([1,2,4,5]<=3)=2/4=0.5 < 0.75 → fail.
        # Step 3 candidates: [1,2,3,4,5,9]
        #   cutoff=4: sens=3/4=0.75 ✓  spec=fraction([3,3,9,9]>4)=2/4=0.5
        #   cutoff=5: sens=4/4=1.0  ✓  spec=fraction([3,3,9,9]>5)=2/4=0.5  ← TIE
        #   cutoff=9: sens=1.0       ✓  spec=0 → worse
        # Tiebreak on equal spec: pick lowest cutoff → 4.0
        err = np.array([1.0, 2.0, 4.0, 5.0])
        ne  = np.array([3.0, 3.0, 9.0, 9.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=0.75)
        assert r['step_used']   == 3
        assert r['sensitivity'] >= 0.75
        assert r['cutoff']      == pytest.approx(4.0)
        assert r['specificity'] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Step 4: No cutoff meets requirement
# ---------------------------------------------------------------------------

class TestStep4EdgeCase:
    def test_no_cutoff_meets_sensitivity(self):
        # Step 4 fires only when min_sensitivity > 1.0 (impossible requirement).
        # With <= convention, cutoff=max(err) always achieves sensitivity=1.0.
        # Use min_sensitivity=1.001 to force step 4.
        err = np.array([1.0, 2.0, 6.0])
        ne  = np.array([5.0, 7.0, 8.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=1.001)
        _check_keys(r)
        _check_counts_consistent(r)
        assert r['step_used']               == 4
        assert r['sensitivity_requirement_met'] is False

    def test_single_error_instance(self):
        err = np.array([3.0])
        ne  = np.array([1.0, 2.0, 4.0, 5.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=0.999)
        _check_keys(r)
        _check_counts_consistent(r)
        assert r['n_error'] == 1


# ---------------------------------------------------------------------------
# Completeness vs contamination direction
# ---------------------------------------------------------------------------

class TestDirectionSymmetry:
    def test_completeness_and_contamination_same_algorithm(self):
        """Same inputs → same numeric outputs, only score_type differs."""
        err = np.array([-3.0, -1.0,  2.0])
        ne  = np.array([ 4.0,  5.0,  6.0])
        rc = compute_completeness_cutoff(err, ne, 0.9)
        rt = compute_contamination_cutoff(err, ne, 0.9)
        assert rc['cutoff']      == pytest.approx(rt['cutoff'])
        assert rc['sensitivity'] == pytest.approx(rt['sensitivity'])
        assert rc['specificity'] == pytest.approx(rt['specificity'])
        assert rc['score_type']  == 'completeness'
        assert rt['score_type']  == 'contamination'

    def test_different_inputs_give_different_cutoffs(self):
        err_comp = np.array([-4.0, -2.0,  1.0])   # completeness: lower scores
        err_cont = np.array([-6.0, -5.0, -3.0])   # contamination: even lower
        ne = np.array([3.0, 4.0, 5.0])
        rc = compute_completeness_cutoff(err_comp, ne, 0.9)
        rt = compute_contamination_cutoff(err_cont, ne, 0.9)
        assert rc['cutoff'] != rt['cutoff']


# ---------------------------------------------------------------------------
# All-errors-identical value
# ---------------------------------------------------------------------------

class TestIdenticalScores:
    def test_all_errors_same_value(self):
        err = np.array([3.0, 3.0, 3.0, 3.0])
        ne  = np.array([5.0, 6.0, 7.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=0.999)
        _check_keys(r)
        _check_counts_consistent(r)
        assert r['sensitivity'] == pytest.approx(1.0)
        assert r['specificity'] == pytest.approx(1.0)

    def test_all_scores_identical_overlap(self):
        err = np.array([2.0, 2.0, 2.0])
        ne  = np.array([2.0, 2.0, 2.0])
        r   = compute_completeness_cutoff(err, ne, min_sensitivity=0.999)
        _check_keys(r)
        _check_counts_consistent(r)
        # Can't achieve perfect separation; step 4 or lower step with low spec
        assert r['sensitivity_requirement_met'] is True or r['step_used'] == 4


# ---------------------------------------------------------------------------
# Batch function smoke tests
# ---------------------------------------------------------------------------

class TestBatchFunctions:
    def _make_batch_inputs(self, n_genomes=20, n_genes=5, seed=0):
        rng = np.random.default_rng(seed)
        presence_orig = rng.integers(0, 2, (n_genomes, n_genes)).astype(np.float32)
        # Build corrupted presence: flip one random gene per genome
        X_corrupt = presence_orig.copy()
        flip_cols = rng.integers(0, n_genes, n_genomes)
        for i, g in enumerate(flip_cols):
            X_corrupt[i, g] = 1 - X_corrupt[i, g]
        error_mask_all = (X_corrupt != presence_orig)

        sign   = 2 * X_corrupt - 1
        logits = rng.uniform(-3, 3, (n_genomes, n_genes)).astype(np.float32)
        scores = logits * sign

        # Completeness: present→absent flips
        comp_mask = error_mask_all & (presence_orig == 1)
        # Contamination: absent→present flips
        cont_mask = error_mask_all & (presence_orig == 0)

        gene_names = [f'gene_{i}' for i in range(n_genes)]
        return scores, comp_mask, cont_mask, presence_orig, gene_names

    def test_completeness_returns_dataframe(self):
        scores, comp_mask, _, pres, names = self._make_batch_inputs()
        df = compute_all_completeness_cutoffs(
            scores, comp_mask, pres, names, min_errors=1)
        assert len(df) == 5
        assert 'gene_name' in df.columns
        assert 'n_completeness_errors' in df.columns
        assert 'cutoff' in df.columns

    def test_contamination_returns_dataframe(self):
        scores, _, cont_mask, pres, names = self._make_batch_inputs()
        df = compute_all_contamination_cutoffs(
            scores, cont_mask, pres, names, min_errors=1)
        assert len(df) == 5
        assert 'n_contamination_errors' in df.columns

    def test_min_errors_skips_genes(self):
        scores, comp_mask, _, pres, names = self._make_batch_inputs(n_genomes=10)
        df = compute_all_completeness_cutoffs(
            scores, comp_mask, pres, names, min_errors=100)
        # With n_genomes=10, no gene can have 100 errors → all NaN
        assert df['cutoff'].isna().all()

    def test_score_type_column(self):
        scores, comp_mask, cont_mask, pres, names = self._make_batch_inputs()
        df_c = compute_all_completeness_cutoffs(scores, comp_mask, pres, names, min_errors=1)
        df_t = compute_all_contamination_cutoffs(scores, cont_mask, pres, names, min_errors=1)
        valid_c = df_c['score_type'].dropna()
        valid_t = df_t['score_type'].dropna()
        assert (valid_c == 'completeness').all()
        assert (valid_t == 'contamination').all()


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    result = pytest.main([__file__, '-v', '--tb=short'])
    sys.exit(result)
