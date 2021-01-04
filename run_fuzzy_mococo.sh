#!/bin/bash
env_name="mc_a"
subspecies_tags="2,2 2,3 3,2 3,3 3,4 4,3 4,4 4,5 5,4 5,5"
ie_and_type="min"
ie_or_type="max"
ie_agg_type="wavg"
min_complexity=2
lv_pop_size=1000
rb_pop_size=2000
rb_p_unspec_init=0.1
num_gens=100
num_collabrs=2
tourn_size=2
lv_p_cross_line=0.75
lv_mut_sigma=0.02
rb_p_cross_swap=0.25
rb_p_mut_flip=0.1

for seed in {0..0}; do
   sbatch fuzzy_mococo.sh \
        "$seed" \
        "$env_name" \
        "$subspecies_tags" \
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
