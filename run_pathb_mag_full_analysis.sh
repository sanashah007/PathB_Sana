#!/bin/bash
#SBATCH --job-name=pathb_analysis
#SBATCH --partition=pi_tami
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/analysis_pipeline_%j.out
#SBATCH --error=logs/analysis_pipeline_%j.err

# Run all PathB metagenome analysis scripts in dependency order.
# Submit from PATHB_sample root:
#   sbatch run_analysis_pipeline.sh

set -euo pipefail

cd /home/sanashah/PATHB_sample

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PathB
export PYTHONUNBUFFERED=1

run() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    shift
    python "$@"
}

# ------------------------------------------------------------------
# 1. Specificity cutoffs — pure numpy/pandas on cached perturbation
#    scores from population_perturbation.py; no model inference.
#    Must run before step 4 (true_disagreements reads its output).
# ------------------------------------------------------------------
run "apply_specificity_cutoffs (k=10000)" \
    population_perturbation/apply_specificity_cutoffs.py --k 10000

# ------------------------------------------------------------------
# 2. Completeness / contamination scores
#    Reads MAG + reference logit caches; independent of step 3.
# ------------------------------------------------------------------
run "completeness_contamination_scores" \
    metagenome_tests/completeness_contamination_scores.py

# ------------------------------------------------------------------
# 3. Raw disagreement counts
#    Reads same caches; independent of step 2.
#    Populates per_genome_summary.tsv for step 6.
# ------------------------------------------------------------------
run "compute_disagreement_metrics" \
    metagenome_tests/adding_disagreements_tami/compute_disagreement_metrics.py

# ------------------------------------------------------------------
# 4. True disagreements (per-gene specificity cutoffs)
#    Depends on step 1 output (completeness_cutoffs.tsv /
#    contamination_cutoffs.tsv).
# ------------------------------------------------------------------
run "true_disagreements" \
    metagenome_tests/true_disagreements.py

# ------------------------------------------------------------------
# 5. Plots — completeness / contamination
#    Depends on step 2 output.
# ------------------------------------------------------------------
run "plot_completeness_contamination" \
    metagenome_tests/plot_completeness_contamination.py

run "plot_score_comparisons (v=1 violin / histogram / boxwhisker)" \
    metagenome_tests/plot_score_comparisons.py

# ------------------------------------------------------------------
# 6. Plots — raw disagreements
#    Depends on step 3 output.
# ------------------------------------------------------------------
run "plot_disagreements (raw, box-and-whisker)" \
    metagenome_tests/adding_disagreements_tami/plot_disagreements.py

# ------------------------------------------------------------------
# 7. Plots — baseline-adjusted disagreements
#    Reads caches directly; no dependency on any compute step.
# ------------------------------------------------------------------
run "plot_disagreements_baseline (box-and-whisker + histogram)" \
    metagenome_tests/adding_disagreements_tami/plot_disagreements_baseline.py

echo ""
echo "============================================================"
echo "  All steps complete."
echo "============================================================"
