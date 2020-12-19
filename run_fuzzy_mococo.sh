#!/bin/bash

and_type="min"
or_type="max"
agg_type="bsum"
num_pos_membership_funcs=4
num_vel_membership_funcs=4
lv_pop_size=40
rb_pop_size=160
num_lv_collbars=4
num_rb_collbars=16
num_lv_elites=2
num_rb_elites=8
num_gens=50
lv_tourn_size=2
rb_tourn_size=8
p_cross_line=0.5
sigma=0.01
p_cross_swap=$(bc -l <<< '1/16')
p_mut_flip=$(bc -l <<< '1/16')
num_perf_rollouts=30
alpha=50.0

for ga_seed in {0..0}; do
    echo sbatch mc_fuzzy_coco.sh \
        "$and_type" "$or_type" "$agg_type" \
        "$num_pos_membership_funcs" \
        "$num_vel_membership_funcs" \
        "$ga_seed" "$lv_pop_size" "$rb_pop_size" \
        "$num_lv_collbars" "$num_rb_collbars" \
        "$num_lv_elites" "$num_rb_elites" \
        "$num_gens" "$lv_tourn_size" "$rb_tourn_size" \
        "$p_cross_line" "$sigma" "$p_cross_swap" \
        "$p_mut_flip" "$num_perf_rollouts" "$alpha"
done
