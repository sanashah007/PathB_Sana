#!/bin/bash
#SBATCH --job-name=cluster_meta
#SBATCH --array=0-142
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cluster_meta_%A_%a.out
#SBATCH --error=logs/cluster_meta_%A_%a.err

# -----------------------------------------------------------------------
# CONFIGURE THESE
# -----------------------------------------------------------------------
FASTA_DIR="/home/sanashah/PATHB_sample/metagenome_tests/ncbi_dataset/ncbi_dataset/data"
OUTPUT_DIR="/home/sanashah/PATHB_sample/metagenome_tests/metagenome_vectors"
LOG_DIR="/home/sanashah/PATHB_sample/logs"
# -----------------------------------------------------------------------

cd /home/sanashah/PATHB_sample

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PathB
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Build a stable sorted list of all FASTAs (same order in every task)
mapfile -t FASTAS < <(find "$FASTA_DIR" -maxdepth 2 \
    \( -name "*.fna" -o -name "*.fa" -o -name "*.fasta" \) | sort)

if [ "${#FASTAS[@]}" -eq 0 ]; then
    echo "ERROR: no FASTA files found in $FASTA_DIR" >&2
    exit 1
fi

FASTA="${FASTAS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$FASTA" ]; then
    echo "ERROR: no FASTA for task index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID -> $FASTA"

# Symlink this genome into a temp dir so the script sees only one FASTA
TMPDIR_GENOME=$(mktemp -d)
ln -s "$(realpath "$FASTA")" "$TMPDIR_GENOME/"

python /home/sanashah/PATHB_sample/metagenome_tests/cluster_metagenomes.py \
    --fasta-dir  "$TMPDIR_GENOME" \
    --output-dir "$OUTPUT_DIR" \
    --log-file   "${LOG_DIR}/assignments_${SLURM_ARRAY_TASK_ID}.tsv" \
    --threads    "$SLURM_CPUS_PER_TASK"

EXIT_CODE=$?
rm -rf "$TMPDIR_GENOME"

# -----------------------------------------------------------------------
# After ALL tasks finish, merge logs with:
#   head -1 logs/assignments_0.tsv > assignments.tsv
#   tail -n +2 -q logs/assignments_*.tsv >> assignments.tsv
# -----------------------------------------------------------------------

exit $EXIT_CODE
