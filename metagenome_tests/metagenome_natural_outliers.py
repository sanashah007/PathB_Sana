#!/usr/bin/env python3
"""
metagenome_natural_outliers.py
==============================
Identifies genes where the PathB model's prediction contradicts the actual
input presence value in MAG assemblies, then compares these disagreements
against each MAG's closest matched reference genome.

Runs at all three variation levels (num_not_shared = 1, 10, 100) so results
can be compared across variation settings.

Steps:
  1. Load assignments.tsv → paired (MAG path, reference path) list
  2. For each variation level, run PathB on all unique genomes in one batch;
     cache per-genome logit arrays so the script can be resumed if interrupted
  3. Flag model-input disagreements per gene per genome per variation
  4. Paired comparison per MAG/reference pair per variation
  5. McNemar's test + BH FDR per gene per variation; join DOR metadata; write outputs

Statistical method:
  The analysis uses paired observations: each MAG is matched to its closest
  reference genome. For gene g, the relevant counts per variation are:
    b = pairs where MAG disagrees, reference does not  (mag_only)
    c = pairs where reference disagrees, MAG does not  (ref_only)
  McNemar's test asks: under H0, b ~ Binomial(b+c, 0.5).  Two-sided exact
  binomial p-values are then corrected for multiple testing with
  Benjamini-Hochberg FDR across all 5,598 genes.  A gene is classified
  mag_specific (or ref_specific) when q < FDR_THRESHOLD AND b > c (or c > b).

Optional quantile sensitivity analysis (--extra-quantiles):
  At variation=1 the model nearly echoes the input, so the logit=0 threshold
  produces zero disagreements.  This analysis uses per-gene lower-tail
  thresholds calibrated from the training set to detect subtler disagreements:
    cutoff[g] = np.percentile(training_v1_scores_g, (1 - q) * 100)
  where training_v1_scores_g are the signed logit (logit × sign) scores for
  gene g across 54 cached unperturbed training-set reference genomes at v=1.
  Because these training scores are mostly near +11.5 (model confident), the
  per-gene minimum is typically >0, making the cutoff more sensitive than
  logit=0 for genes where the model is usually very confident.
  Outputs to: metagenome_natural_outliers/quantile_sensitivity/

Must be run from PATHB_sample root where run_model.py and twolayer.pth reside.

Outputs (in --out-dir):
  per_gene_summary.tsv   — one row per gene × variation; columns include
                           variation, rates, p_mcnemar, q_fdr, classification,
                           direction, DOR metadata
  per_pair_summary.tsv   — one row per MAG/reference pair × variation
  summary.txt            — human-readable overview; top genes per variation
  cache/                 — cached logit arrays ({stem}_v{V}.npy per genome)
  quantile_sensitivity/  — extra outputs from --extra-quantiles (if given)
    var1_quantile_comparison.tsv
    var1_quantile_summary.txt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_model as _rm

G             = 5598
VARIATIONS    = [1, 10, 100]
FDR_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for McNemar's test."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(b, n, p=0.5, alternative='two-sided').pvalue


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values clamped to [0,1]."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p_values * n / ranks
    q_mono = np.minimum.accumulate(q[order[::-1]])[::-1]
    q_final = np.empty(n)
    q_final[order] = q_mono
    return np.minimum(q_final, 1.0)


# ---------------------------------------------------------------------------
# Quantile sensitivity analysis (variation=1 only)
# ---------------------------------------------------------------------------

def run_quantile_sensitivity(
    extra_quantiles: list[float],
    pairs: list,
    logit_arr_v1: dict,
    pres_arr: dict,
    gene_names: list,
    dor_df: pd.DataFrame,
    out_dir: Path,
    cache_dir: Path,
    ref_dir: Path,
    vectors_dir: Path,
    n_pairs: int,
) -> None:
    """
    Per-gene quantile sensitivity analysis at variation=1.

    For each extra quantile q, the per-gene cutoff is the (1-q)th percentile
    of signed logit scores from unperturbed training-set reference genomes
    cached at variation=1:
        cutoff[g] = np.percentile(training_scores_g, (1 - q) * 100)

    A genome's gene g is flagged as disagreeing when its signed logit score
    falls below cutoff[g].  Since most training scores are near +11.5 (model
    near-certain at variation=1), the per-gene minimum is typically >0, making
    this threshold more sensitive than the default logit=0 criterion.
    """
    qs_dir = out_dir / 'quantile_sensitivity'
    qs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Use all cached genomes (MAGs + references) to build the per-gene
    # score distribution.  Train/val/test split is irrelevant — we are not
    # evaluating model performance.  Including MAGs does not bias the
    # McNemar's test because the cutoff is a fixed threshold applied equally
    # to both sides of each pair; asymmetry between pairs is what the test
    # measures, and that is unaffected by the absolute level of the cutoff.
    # ------------------------------------------------------------------
    print('\nQuantile sensitivity: loading all cached v1 scores ...')

    all_cached = sorted(
        fn[:-7]  # strip '_v1.npy'
        for fn in os.listdir(str(cache_dir))
        if fn.endswith('_v1.npy')
    )
    print(f'  Genomes in v1 cache: {len(all_cached)}')

    # Collect per-gene signed logit (error) scores from training references
    # Shape after collection: scores_per_gene[g] = list of floats
    scores_per_gene: list[list] = [[] for _ in range(G)]
    n_loaded = 0
    for stem in all_cached:
        # Presence file may be in ref_dir (references) or vectors_dir (MAGs)
        pres_path = ref_dir / f'{stem}.txt'
        if not pres_path.exists():
            pres_path = vectors_dir / f'{stem}.txt'
        if not pres_path.exists():
            continue
        logits = np.load(str(cache_dir / f'{stem}_v1.npy')).astype(np.float64)
        pres   = load_presence(pres_path).astype(np.float64)
        escore = logits * (2.0 * pres - 1.0)
        for g in range(G):
            scores_per_gene[g].append(escore[g])
        n_loaded += 1

    print(f'  Error scores collected from {n_loaded} genomes × {G} genes')

    # ------------------------------------------------------------------
    # Compute per-gene cutoffs at each extra quantile
    # ------------------------------------------------------------------
    # cutoffs[q][g] = threshold below which gene g is flagged as disagreeing
    cutoffs: dict[float, np.ndarray] = {}
    for q in extra_quantiles:
        alpha = 1.0 - q          # lower-tail fraction flagged in training
        pct   = alpha * 100.0    # percentile rank
        c_arr = np.array([
            np.percentile(scores_per_gene[g], pct) for g in range(G)
        ], dtype=np.float64)
        cutoffs[q] = c_arr
        pos = (c_arr > 0).sum()
        print(f'  q={q}: cutoff percentile rank={pct:.3f}  '
              f'genes with cutoff>0: {pos}/{G}  '
              f'median cutoff: {np.median(c_arr):.3f}')

    # ------------------------------------------------------------------
    # Apply cutoffs → disagreement matrices at v=1
    # ------------------------------------------------------------------
    # disagree_q[q][stem] = bool (G,) array
    disagree_q: dict[float, dict[str, np.ndarray]] = {}
    for q in extra_quantiles:
        c_arr = cutoffs[q]
        disagree_q[q] = {}
        for stem, pres in pres_arr.items():
            logits  = logit_arr_v1[stem].astype(np.float64)
            escore  = logits * (2.0 * pres.astype(np.float64) - 1.0)
            disagree_q[q][stem] = escore < c_arr

    # ------------------------------------------------------------------
    # Paired comparison + McNemar's test + BH FDR for each extra quantile
    # ------------------------------------------------------------------
    # Accumulate per-gene (b, c) = (mag_only, ref_only) counts
    bc: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for q in extra_quantiles:
        b_arr = np.zeros(G, dtype=np.int32)
        c_arr = np.zeros(G, dtype=np.int32)
        for mag_p, ref_p, mag_stem, ref_stem in pairs:
            d_mag = disagree_q[q][mag_stem]
            d_ref = disagree_q[q][ref_stem]
            b_arr += (d_mag & ~d_ref).astype(np.int32)
            c_arr += (d_ref & ~d_mag).astype(np.int32)
        bc[q] = (b_arr, c_arr)

    # McNemar p-values + BH FDR per quantile
    results: dict[float, dict] = {}
    for q in extra_quantiles:
        b_arr, c_arr = bc[q]
        p_vals = np.array([
            mcnemar_pvalue(int(b_arr[g]), int(c_arr[g]))
            for g in range(G)
        ])
        q_vals = bh_fdr(p_vals)
        results[q] = {
            'b': b_arr, 'c': c_arr, 'p': p_vals, 'q': q_vals,
        }

    # ------------------------------------------------------------------
    # Build wide-format comparison TSV (one row per gene)
    # ------------------------------------------------------------------
    dor_idx = dor_df.set_index('gene_idx')[
        ['category', 'log10_dor', 'sensitivity', 'specificity']
    ]

    rows = []
    for g in range(G):
        if g in dor_idx.index:
            meta = dor_idx.loc[g]
            cat, dor, sens, spec = (meta['category'], float(meta['log10_dor']),
                                    float(meta['sensitivity']), float(meta['specificity']))
        else:
            cat, dor, sens, spec = '', float('nan'), float('nan'), float('nan')

        row = {
            'gene_idx':    g,
            'gene_name':   gene_names[g],
            'category':    cat,
            'log10_dor':   dor,
            'sensitivity': sens,
            'specificity': spec,
        }

        for q in extra_quantiles:
            b  = int(results[q]['b'][g])
            c  = int(results[q]['c'][g])
            qv = float(results[q]['q'][g])
            p  = float(results[q]['p'][g])
            rm = b / n_pairs
            rr = c / n_pairs

            if   qv < FDR_THRESHOLD and b > c: cls = 'mag_specific'
            elif qv < FDR_THRESHOLD and c > b: cls = 'ref_specific'
            else:                               cls = 'neither'

            tag = f'q{q}'
            row[f'cutoff_{tag}']        = float(cutoffs[q][g])
            row[f'n_mag_only_{tag}']    = b
            row[f'n_ref_only_{tag}']    = c
            row[f'rate_mag_only_{tag}'] = rm
            row[f'rate_ref_only_{tag}'] = rr
            row[f'p_mcnemar_{tag}']     = p
            row[f'q_fdr_{tag}']         = qv
            row[f'classification_{tag}']= cls

        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(qs_dir / 'var1_quantile_comparison.tsv', sep='\t', index=False)

    # ------------------------------------------------------------------
    # Write human-readable summary
    # ------------------------------------------------------------------
    def dor_tier(v):
        if pd.isna(v): return 'no_dor'
        if v >= 2:     return 'high'
        if v >= 0.5:   return 'moderate'
        return 'low'

    with open(qs_dir / 'var1_quantile_summary.txt', 'w') as fh:
        fh.write('Quantile Sensitivity Analysis — variation=1\n')
        fh.write('============================================\n\n')
        fh.write(f'Genomes used for cutoffs:              {n_loaded} (MAGs + references)\n')
        fh.write(f'MAG/reference pairs:                   {n_pairs}\n')
        fh.write(f'Extra quantiles tested:                {extra_quantiles}\n')
        fh.write(f'FDR threshold:                         {FDR_THRESHOLD}\n\n')
        fh.write('Cutoff semantics:\n')
        fh.write('  cutoff[g] = np.percentile(training_v1_scores_g, (1-q)*100)\n')
        fh.write('  A gene is flagged when error_score = logit*sign < cutoff[g]\n')
        fh.write('  Larger q → lower percentile rank → cutoff near training minimum\n')
        fh.write('  → more sensitive than logit=0 for genes where training min > 0\n\n')

        for q in extra_quantiles:
            tag = f'q{q}'
            sub = comp_df[['gene_idx', 'gene_name', 'category', 'log10_dor',
                            f'cutoff_{tag}',
                            f'n_mag_only_{tag}', f'n_ref_only_{tag}',
                            f'rate_mag_only_{tag}', f'rate_ref_only_{tag}',
                            f'p_mcnemar_{tag}', f'q_fdr_{tag}',
                            f'classification_{tag}']].copy()
            sub.columns = ['gene_idx', 'gene_name', 'category', 'log10_dor',
                           'cutoff', 'n_mag_only', 'n_ref_only',
                           'rate_mag_only', 'rate_ref_only',
                           'p_mcnemar', 'q_fdr', 'classification']

            mag_spec = sub[sub['classification'] == 'mag_specific'].copy()
            ref_spec = sub[sub['classification'] == 'ref_specific'].copy()
            cls_cts  = sub['classification'].value_counts()

            fh.write(f'{"="*60}\n')
            fh.write(f'QUANTILE = {q}  '
                     f'(cutoff percentile rank = {(1-q)*100:.4f}%)\n')
            fh.write(f'{"="*60}\n\n')

            fh.write('Cutoff statistics (across 5598 genes):\n')
            cutoffs_arr = comp_df[f'cutoff_{tag}'].values
            fh.write(f'  min={cutoffs_arr.min():.3f}  '
                     f'median={np.median(cutoffs_arr):.3f}  '
                     f'max={cutoffs_arr.max():.3f}\n')
            fh.write(f'  Genes with cutoff > 0: {(cutoffs_arr > 0).sum()}/{G}\n\n')

            fh.write('Gene classification counts (FDR q < 0.05):\n')
            for cls in ('mag_specific', 'ref_specific', 'neither'):
                fh.write(f'  {cls:<20s}: {cls_cts.get(cls, 0)}\n')

            if not mag_spec.empty:
                mag_spec['_tier'] = mag_spec['log10_dor'].apply(dor_tier)
                n_high = (mag_spec['_tier'] == 'high').sum()
                fh.write(f'\nMAG-specific: {len(mag_spec)} genes  '
                         f'({n_high} with high DOR = {n_high/len(mag_spec):.1%})\n')

            fh.write('\n--- Top 10 MAG-specific genes (by q_fdr asc) ---\n')
            fh.write(f'{"gene_name":<62} {"cat":<12} {"log10_dor":>10} '
                     f'{"q_fdr":>8} {"rate_mag":>9} {"rate_ref":>9} cutoff\n')
            top = mag_spec.sort_values('q_fdr').head(10)
            for _, row in top.iterrows():
                dv = f'{row["log10_dor"]:.3f}' if not pd.isna(row['log10_dor']) else 'N/A'
                fh.write(f'{str(row["gene_name"])[:61]:<62} {str(row["category"]):<12} '
                         f'{dv:>10} {row["q_fdr"]:>8.2e} '
                         f'{row["rate_mag_only"]:>9.3f} {row["rate_ref_only"]:>9.3f} '
                         f'{row["cutoff"]:>7.3f}\n')

            fh.write('\n--- Top 10 REF-specific genes (by q_fdr asc) ---\n')
            fh.write(f'{"gene_name":<62} {"cat":<12} {"log10_dor":>10} '
                     f'{"q_fdr":>8} {"rate_mag":>9} {"rate_ref":>9} cutoff\n')
            top = ref_spec.sort_values('q_fdr').head(10)
            for _, row in top.iterrows():
                dv = f'{row["log10_dor"]:.3f}' if not pd.isna(row['log10_dor']) else 'N/A'
                fh.write(f'{str(row["gene_name"])[:61]:<62} {str(row["category"]):<12} '
                         f'{dv:>10} {row["q_fdr"]:>8.2e} '
                         f'{row["rate_mag_only"]:>9.3f} {row["rate_ref_only"]:>9.3f} '
                         f'{row["cutoff"]:>7.3f}\n')
            fh.write('\n')

    print(f'\nQuantile sensitivity outputs written to: {qs_dir}')
    print(f'  var1_quantile_comparison.tsv — {len(comp_df)} genes × {len(extra_quantiles)} quantiles')
    print(f'  var1_quantile_summary.txt')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Identify PathB model-input disagreements in MAGs vs matched references.'
    )
    p.add_argument('--vectors-dir',   default='metagenome_tests/metagenome_vectors/')
    p.add_argument('--assignments',   default='metagenome_tests/assignments.tsv')
    p.add_argument('--ref-dir',       default='all_embeded_genomes/')
    p.add_argument('--dor',
                   default='population_perturbation/population_perturbation_results/per_gene_dor.tsv')
    p.add_argument('--genes',         default='gene_names.txt')
    p.add_argument('--out-dir',       default='metagenome_tests/metagenome_natural_outliers/')
    p.add_argument('--extra-quantiles',
                   default='0.999,0.9999',
                   help='Comma-separated quantile levels for sensitivity analysis at '
                        'variation=1.  For each q, uses the (1-q)th percentile of '
                        'cached training-set variation=1 error scores as a per-gene '
                        'cutoff.  Pass empty string to skip.')
    args = p.parse_args()

    out_dir   = Path(args.out_dir)
    cache_dir = out_dir / 'cache'
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    gene_names = [ln.rstrip('\n\r') for ln in open(args.genes)]
    if len(gene_names) != G:
        sys.exit(f'gene_names has {len(gene_names)} entries, expected {G}')

    extra_quantiles: list[float] = []
    if args.extra_quantiles.strip():
        for s in args.extra_quantiles.strip().split(','):
            try:
                q = float(s.strip())
                if not 0 < q < 1:
                    sys.exit(f'--extra-quantiles values must be in (0, 1), got {q}')
                extra_quantiles.append(q)
            except ValueError:
                sys.exit(f'--extra-quantiles: cannot parse {s!r} as float')

    # -----------------------------------------------------------------------
    # Step 1 — Build paired list from assignments.tsv
    # -----------------------------------------------------------------------
    print('Step 1: Building MAG/reference pairs ...')
    asgn = pd.read_csv(args.assignments, sep='\t')
    for col in asgn.select_dtypes('object').columns:
        asgn[col] = asgn[col].str.strip()

    vectors_dir = Path(args.vectors_dir)
    ref_dir     = Path(args.ref_dir)

    pairs     = []
    not_found = []

    for _, row in asgn.iterrows():
        cr = str(row.get('closest_ref', '')).strip()
        if not cr or cr in ('N/A', 'nan'):
            not_found.append((str(row.get('output_file', '')), 'no closest_ref'))
            continue

        mag_filename = str(row['output_file']).strip()
        mag_path     = vectors_dir / mag_filename
        ref_path     = ref_dir / f'{cr}.txt'

        if not mag_path.exists():
            not_found.append((mag_filename, 'MAG vector not found'))
            continue
        if not ref_path.exists():
            not_found.append((mag_filename, f'reference not found: {cr}.txt'))
            continue

        pairs.append((mag_path, ref_path, mag_path.stem, cr))

    print(f'  {len(pairs)} valid pairs')
    if not_found:
        print(f'  {len(not_found)} skipped:')
        for name, reason in not_found:
            print(f'    {name}: {reason}')
    if not pairs:
        sys.exit('No valid pairs found.')

    # -----------------------------------------------------------------------
    # Step 2 — Run PathB at each variation level (batch + resume-safe cache)
    # -----------------------------------------------------------------------
    unique_stems: dict[str, Path] = {}
    for mag_p, ref_p, mag_stem, ref_stem in pairs:
        unique_stems.setdefault(mag_stem, mag_p)
        unique_stems.setdefault(ref_stem, ref_p)

    print(f'\nStep 2: Running PathB on {len(unique_stems)} unique genomes '
          f'at variations {VARIATIONS} ...')

    logit_arr: dict[int, dict[str, np.ndarray]] = {}

    for v in VARIATIONS:
        stems_to_run = [
            (stem, path) for stem, path in unique_stems.items()
            if not (cache_dir / f'{stem}_v{v}.npy').exists()
        ]
        n_cached = len(unique_stems) - len(stems_to_run)
        print(f'  variation={v}: {n_cached} cached, {len(stems_to_run)} to run')

        if stems_to_run:
            run_stems, run_paths = zip(*stems_to_run)
            X = np.stack([load_presence(p) for p in run_paths], axis=0)
            batch_logits = run_model_batch(X, num_not_shared=v)
            for i, stem in enumerate(run_stems):
                np.save(str(cache_dir / f'{stem}_v{v}.npy'),
                        batch_logits[i].astype(np.float32))

        logit_arr[v] = {
            stem: np.load(str(cache_dir / f'{stem}_v{v}.npy'))
            for stem in unique_stems
        }

    pres_arr: dict[str, np.ndarray] = {
        stem: load_presence(path) for stem, path in unique_stems.items()
    }

    # -----------------------------------------------------------------------
    # Step 3 — Flag model-input disagreements (logit=0 threshold)
    # -----------------------------------------------------------------------
    print('\nStep 3: Flagging model-input disagreements ...')
    # 0 = no disagreement
    # 1 = absent_predicted_present  (pres==0, logit>0)
    # 2 = present_predicted_absent  (pres==1, logit<0)
    disagree: dict[int, dict[str, np.ndarray]] = {}
    for v in VARIATIONS:
        disagree[v] = {}
        for stem in unique_stems:
            pres   = pres_arr[stem]
            logits = logit_arr[v][stem]
            d = np.zeros(G, dtype=np.int8)
            d[(pres == 0) & (logits > 0)] = 1
            d[(pres == 1) & (logits < 0)] = 2
            disagree[v][stem] = d

    # -----------------------------------------------------------------------
    # Step 4 — Paired comparison per variation
    # -----------------------------------------------------------------------
    print('Step 4: Paired comparison ...')

    cnt: dict[int, dict[str, np.ndarray]] = {
        v: {k: np.zeros(G, dtype=np.int32)
            for k in ('mag_only', 'ref_only', 'both', 'neither')}
        for v in VARIATIONS
    }
    pair_rows: dict[int, list] = {v: [] for v in VARIATIONS}

    for v in VARIATIONS:
        for mag_p, ref_p, mag_stem, ref_stem in pairs:
            d_mag = disagree[v][mag_stem] > 0
            d_ref = disagree[v][ref_stem] > 0

            mag_only = d_mag & ~d_ref
            ref_only = d_ref & ~d_mag
            both     = d_mag &  d_ref
            neither  = ~d_mag & ~d_ref

            cnt[v]['mag_only'] += mag_only.astype(np.int32)
            cnt[v]['ref_only'] += ref_only.astype(np.int32)
            cnt[v]['both']     += both.astype(np.int32)
            cnt[v]['neither']  += neither.astype(np.int32)

            pair_rows[v].append({
                'variation':          v,
                'mag_filename':       mag_stem,
                'reference_filename': ref_stem,
                'n_mag_only':         int(mag_only.sum()),
                'n_ref_only':         int(ref_only.sum()),
                'n_both':             int(both.sum()),
                'n_neither':          int(neither.sum()),
                '_mag_only_idx':      np.where(mag_only)[0].tolist(),
            })

    # -----------------------------------------------------------------------
    # Step 5 — McNemar's test + BH FDR; aggregate per gene per variation
    # -----------------------------------------------------------------------
    print('Step 5: McNemar\'s test + BH FDR correction ...')

    n_pairs   = len(pairs)
    mag_stems = [mag_stem for _, _, mag_stem, _ in pairs]

    dir_ap: dict[int, np.ndarray] = {}
    dir_pa: dict[int, np.ndarray] = {}
    for v in VARIATIONS:
        ap = np.zeros(G, dtype=np.int32)
        pa = np.zeros(G, dtype=np.int32)
        for stem in mag_stems:
            d = disagree[v][stem]
            ap += (d == 1).astype(np.int32)
            pa += (d == 2).astype(np.int32)
        dir_ap[v] = ap
        dir_pa[v] = pa

    dor_df  = pd.read_csv(args.dor, sep='\t')
    dor_idx = dor_df.set_index('gene_idx')[
        ['category', 'log10_dor', 'sensitivity', 'specificity']
    ]

    gene_rows = []

    for v in VARIATIONS:
        b_arr = cnt[v]['mag_only']
        c_arr = cnt[v]['ref_only']

        p_values = np.array([
            mcnemar_pvalue(int(b_arr[g]), int(c_arr[g]))
            for g in range(G)
        ])
        q_values = bh_fdr(p_values)

        rate_mag   = b_arr / n_pairs
        rate_ref   = c_arr / n_pairs
        rate_both  = cnt[v]['both'] / n_pairs
        rate_delta = rate_mag - rate_ref

        for g in range(G):
            if g in dor_idx.index:
                meta = dor_idx.loc[g]
                cat, dor, sens, spec = (meta['category'], float(meta['log10_dor']),
                                        float(meta['sensitivity']), float(meta['specificity']))
            else:
                cat, dor, sens, spec = '', float('nan'), float('nan'), float('nan')

            b, c = int(b_arr[g]), int(c_arr[g])
            q    = float(q_values[g])

            if   q < FDR_THRESHOLD and b > c: cls = 'mag_specific'
            elif q < FDR_THRESHOLD and c > b: cls = 'ref_specific'
            else:                              cls = 'neither'

            ap_g, pa_g = int(dir_ap[v][g]), int(dir_pa[v][g])
            if   ap_g > 0 and pa_g == 0: direction = 'absent_predicted_present'
            elif pa_g > 0 and ap_g == 0: direction = 'present_predicted_absent'
            elif ap_g > 0 and pa_g > 0:  direction = 'mixed'
            else:                         direction = 'none'

            gene_rows.append({
                'variation':      v,
                'gene_idx':       g,
                'gene_name':      gene_names[g],
                'category':       cat,
                'log10_dor':      dor,
                'sensitivity':    sens,
                'specificity':    spec,
                'n_mag_only':     b,
                'n_ref_only':     c,
                'n_both':         int(cnt[v]['both'][g]),
                'n_neither':      int(cnt[v]['neither'][g]),
                'rate_mag_only':  float(rate_mag[g]),
                'rate_ref_only':  float(rate_ref[g]),
                'rate_both':      float(rate_both[g]),
                'rate_delta':     float(rate_delta[g]),
                'p_mcnemar':      float(p_values[g]),
                'q_fdr':          q,
                'classification': cls,
                'direction':      direction,
            })

    per_gene_df = pd.DataFrame(gene_rows)

    # -----------------------------------------------------------------------
    # Finalise per-pair rows
    # -----------------------------------------------------------------------
    g_dor_map = dor_df.set_index('gene_idx')['log10_dor'].to_dict()

    pair_out_rows = []
    for v in VARIATIONS:
        for row in pair_rows[v]:
            idx_list = row.pop('_mag_only_idx')
            top5_idx = sorted(
                idx_list,
                key=lambda g: g_dor_map.get(g, float('-inf'))
                              if not np.isnan(g_dor_map.get(g, float('nan'))) else float('-inf'),
                reverse=True,
            )[:5]
            row['top_mag_only_genes'] = ','.join(gene_names[g] for g in top5_idx)
            pair_out_rows.append(row)

    per_pair_df = pd.DataFrame(pair_out_rows)

    # -----------------------------------------------------------------------
    # Write existing outputs (unchanged)
    # -----------------------------------------------------------------------
    print('\nWriting outputs ...')
    per_gene_df.to_csv(out_dir / 'per_gene_summary.tsv',  sep='\t', index=False)
    per_pair_df.to_csv(out_dir / 'per_pair_summary.tsv',  sep='\t', index=False)

    def dor_tier(v):
        if pd.isna(v): return 'no_dor'
        if v >= 2:     return 'high'
        if v >= 0.5:   return 'moderate'
        return 'low'

    with open(out_dir / 'summary.txt', 'w') as fh:
        fh.write('Metagenome Natural Outliers — Summary\n')
        fh.write('======================================\n\n')
        fh.write(f'Total MAG/reference pairs:  {n_pairs}\n')
        fh.write(f'Unique reference genomes:    {len({r for _,_,_,r in pairs})}\n')
        fh.write(f'Genes:                       {G}\n')
        fh.write(f'Variations tested:           {VARIATIONS}\n')
        fh.write(f'Statistical test:            McNemar\'s exact (two-sided binomial)\n')
        fh.write(f'FDR correction:              Benjamini-Hochberg, alpha={FDR_THRESHOLD}\n\n')

        for v in VARIATIONS:
            vdf = per_gene_df[per_gene_df['variation'] == v]
            cls_counts = vdf['classification'].value_counts()
            mag_spec = vdf[vdf['classification'] == 'mag_specific'].copy()
            ref_spec = vdf[vdf['classification'] == 'ref_specific'].copy()

            fh.write(f'{"="*60}\n')
            fh.write(f'VARIATION = {v}\n')
            fh.write(f'{"="*60}\n\n')

            fh.write('Gene classification counts (FDR q < 0.05):\n')
            for cls in ('mag_specific', 'ref_specific', 'neither'):
                fh.write(f'  {cls:<20s}: {cls_counts.get(cls, 0)}\n')

            if not mag_spec.empty:
                fh.write('\nMAG-specific genes by category:\n')
                for cat, grp in mag_spec.groupby('category'):
                    fh.write(f'  {cat:<15s}: {len(grp)}\n')

                mag_spec['_tier'] = mag_spec['log10_dor'].apply(dor_tier)
                fh.write('\nMAG-specific genes by DOR tier:\n')
                for tier, label in [('high','>=2'), ('moderate','0.5-2'),
                                    ('low','<0.5'), ('no_dor','no DOR data')]:
                    n = (mag_spec['_tier'] == tier).sum()
                    fh.write(f'  {label:<15s}: {n}\n')
                n_high = (mag_spec['_tier'] == 'high').sum()
                fh.write(f'Fraction MAG-specific with high DOR: '
                         f'{n_high/len(mag_spec):.1%} ({n_high}/{len(mag_spec)})\n')

            fh.write('\n--- Top 20 MAG-specific genes (by q_fdr asc, then rate_delta desc) ---\n')
            fh.write(f'{"gene_name":<62} {"cat":<12} {"log10_dor":>10} '
                     f'{"q_fdr":>8} {"rate_mag":>9} {"rate_ref":>9} direction\n')
            for _, row in (mag_spec.sort_values(['q_fdr', 'rate_delta'],
                                                ascending=[True, False])
                           .head(20).iterrows()):
                dv = f'{row["log10_dor"]:.3f}' if not pd.isna(row['log10_dor']) else 'N/A'
                fh.write(f'{str(row["gene_name"])[:61]:<62} {str(row["category"]):<12} '
                         f'{dv:>10} {row["q_fdr"]:>8.2e} '
                         f'{row["rate_mag_only"]:>9.3f} {row["rate_ref_only"]:>9.3f} '
                         f'{row["direction"]}\n')

            fh.write('\n--- Top 20 REF-specific genes (by q_fdr asc, then rate_delta asc) ---\n')
            fh.write(f'{"gene_name":<62} {"cat":<12} {"log10_dor":>10} '
                     f'{"q_fdr":>8} {"rate_mag":>9} {"rate_ref":>9} direction\n')
            for _, row in (ref_spec.sort_values(['q_fdr', 'rate_delta'],
                                                ascending=[True, True])
                           .head(20).iterrows()):
                dv = f'{row["log10_dor"]:.3f}' if not pd.isna(row['log10_dor']) else 'N/A'
                fh.write(f'{str(row["gene_name"])[:61]:<62} {str(row["category"]):<12} '
                         f'{dv:>10} {row["q_fdr"]:>8.2e} '
                         f'{row["rate_mag_only"]:>9.3f} {row["rate_ref_only"]:>9.3f} '
                         f'{row["direction"]}\n')
            fh.write('\n')

    n_gene_rows = len(per_gene_df)
    n_pair_rows = len(per_pair_df)
    print(f'\nDone.')
    print(f'  per_gene_summary.tsv  — {n_gene_rows} rows ({G} genes × {len(VARIATIONS)} variations)')
    print(f'  per_pair_summary.tsv  — {n_pair_rows} rows ({n_pairs} pairs × {len(VARIATIONS)} variations)')
    print(f'  summary.txt')
    print(f'  cache/                — {len(unique_stems) * len(VARIATIONS)} cached logit arrays')

    # -----------------------------------------------------------------------
    # Optional: quantile sensitivity analysis at variation=1
    # -----------------------------------------------------------------------
    if extra_quantiles:
        run_quantile_sensitivity(
            extra_quantiles = extra_quantiles,
            pairs           = pairs,
            logit_arr_v1    = logit_arr[1],
            pres_arr        = pres_arr,
            gene_names      = gene_names,
            dor_df          = dor_df,
            out_dir         = out_dir,
            cache_dir       = cache_dir,
            ref_dir         = ref_dir,
            vectors_dir     = vectors_dir,
            n_pairs         = n_pairs,
        )


if __name__ == '__main__':
    main()
