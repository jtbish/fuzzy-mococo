#!/bin/bash
#SBATCH --partition=coursework
#SBATCH --cpus-per-task=8

source ~/virtualenvs/mococo/bin/activate
python3 fuzzy_mococo.py \
    --experiment-name="$SLURM_JOB_ID" \
    --seed="$1" \
    --env-name="$2" \
    --subspecies-tags="$3" \
    --subspecies-pmf-base="$4" \
    --ie-and-type="$5" \
    --ie-or-type="$6" \
    --ie-agg-type="$7" \
    --min-complexity="$8" \
    --lv-pop-size="$9" \
    --rb-pop-size="${10}" \
    --rb-p-unspec-init="${11}" \
    --num-gens="${12}" \
    --num-collabrs="${13}" \
    --tourn-size="${14}" \
    --lv-p-cross-line="${15}" \
    --lv-mut-sigma="${16}" \
    --rb-p-cross-swap="${17}" \
    --rb-p-mut-flip="${18}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
