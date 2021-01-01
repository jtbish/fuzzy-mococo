#!/bin/bash
#SBATCH --partition=coursework
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
    --lv-pop-size="$7" \
    --rb-pop-size="$8" \
    --rb-p-unspec-init="$9" \
    --num-gens="${10}" \
    --num-collabrs="${11}" \
    --lv-tourn-size="${12}" \
    --rb-tourn-size="${13}" \
    --lv-p-cross-line="${14}" \
    --lv-mut-sigma="${15}" \
    --rb-p-cross-swap="${16}" \
    --rb-p-mut-flip="${17}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
