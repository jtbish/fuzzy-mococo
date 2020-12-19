#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8

source ~/virtualenvs/piecewise/bin/activate
python3 mc_fuzzy_coco.py \
    --and-type="$1" --or-type="$2" --agg-type="$3" \
    --num-pos-membership-funcs="$4" --num-vel-membership-funcs="$5" \
    --ga-seed="$6" --lv-pop-size="$7" --rb-pop-size="$8" \
    --num-lv-collabrs="$9" --num-rb-collabrs="${10}" \
    --num-lv-elites="${11}" --num-rb-elites="${12}" \
    --num-gens="${13}" --lv-tourn-size="${14}" --rb-tourn-size="${15}" \
    --p-cross-line="${16}" --sigma="${17}" --p-cross-swap="${18}" \
    --p-mut-flip="${19}" --num-perf-rollouts="${20}" --alpha="${21}" \
    --experiment-name="$SLURM_JOB_ID"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
