#!/bin/bash
env_name="mc_a"
subspecies_tags="3,3 4,4 5,5"
ie_and_type="min"
ie_or_type="max"
ie_agg_type="wavg"
lv_pop_size=500
rb_pop_size=500
rb_p_unspec_init=0.15
num_gens=50
num_collabrs=10
lv_tourn_size=2
rb_tourn_size=2
lv_p_cross_line=0.75
lv_mut_sigma=0.02
rb_cross_swap_mult=2
rb_mut_flip_mult=1

for seed in {0..0}; do
   sbatch fuzzy_mococo.sh \
        "$seed" \
        "$env_name" \
        "$subspecies_tags" \
        "$ie_and_type" \
        "$ie_or_type" \
        "$ie_agg_type" \
        "$lv_pop_size" \
        "$rb_pop_size" \
        "$rb_p_unspec_init" \
        "$num_gens" \
        "$num_collabrs" \
        "$lv_tourn_size" \
        "$rb_tourn_size" \
        "$lv_p_cross_line" \
        "$lv_mut_sigma" \
        "$rb_cross_swap_mult" \
        "$rb_mut_flip_mult"
done
