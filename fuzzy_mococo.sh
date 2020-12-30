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
    --rb-unspec-init-mult="$9" \
    --num-gens="${10}" \
    --num-collabrs="${11}" \
    --lv-tourn-size="${12}" \
    --rb-tourn-size="${13}" \
    --lv-p-cross-line="${14}" \
    --lv-mut-sigma="${15}" \
    --rb-cross-swap-mult="${16}" \
    --rb-mut-flip-mult="${17}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
