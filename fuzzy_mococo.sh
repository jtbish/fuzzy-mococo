#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8

source ~/virtualenvs/mococo/bin/activate
python3 fuzzy_mococo.py \
    --experiment-name="$SLURM_JOB_ID" \
    --seed="$1" \
    --env-name="$2" \
    --subspecies-tags="$3" \
    --ie-and-type="$4" \
    --ie-or-type="$5" \
    --ie-agg-type="$6" \
    --min-complexity="$7" \
    --lv-pop-size="$8" \
    --rb-pop-size="$9" \
    --rb-p-unspec-init="${10}" \
    --num-gens="${11}" \
    --num-collabrs="${12}" \
    --tourn-size="${13}" \
    --lv-p-cross-line="${14}" \
    --lv-mut-sigma="${15}" \
    --rb-p-cross-swap="${16}" \
    --rb-p-mut-flip="${17}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
