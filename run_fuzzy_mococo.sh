#!/bin/bash
ie_and_type="min"
ie_or_type="max"
ie_agg_type="wavg"
min_complexity=2
rb_p_unspec_init=0.1
num_gens=50
num_collabrs=2
tourn_size=2
lv_p_cross_line=0.75
lv_mut_sigma=0.02
rb_p_cross_swap=0.25
rb_p_mut_flip=0.05

env_name="cp_b"
subspecies_tags="2,2,2,2 3,2,3,2 3,3,3,3"
subspecies_pmf_base=1.025
lv_pop_size=400
rb_pop_size=2000

for seed in {25..29}; do
   echo sbatch fuzzy_mococo.sh \
        "$seed" \
        "$env_name" \
        "$subspecies_tags" \
        "$subspecies_pmf_base" \
        "$ie_and_type" \
        "$ie_or_type" \
        "$ie_agg_type" \
        "$min_complexity" \
        "$lv_pop_size" \
        "$rb_pop_size" \
        "$rb_p_unspec_init" \
        "$num_gens" \
        "$num_collabrs" \
        "$tourn_size" \
        "$lv_p_cross_line" \
        "$lv_mut_sigma" \
        "$rb_p_cross_swap" \
        "$rb_p_mut_flip"
done
