import argparse
import numpy as np
from rlenvs.cartpole import make_cartpole_a_env as make_cp
import itertools

MAX_VAL = 200.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-v-npy", required=True)
    return parser.parse_args()


def main(args):
    v = np.load(args.cp_v_npy)
    equal_num_bins_on_each_dim = (len(set(v.shape)) == 1)
    assert equal_num_bins_on_each_dim
    num_bins_per_dim = v.shape[0]

    env = make_cp()

    assert num_bins_per_dim % 2 == 0

    idx_left = ((num_bins_per_dim // 2) - 1)
    idx_right = (num_bins_per_dim // 2)
    idx_left_min = 0
    idx_right_max = (num_bins_per_dim - 1)

    # successively grow a hypercube over the idxs of v (equating to a
    # hyperrectangle in the discretised feature space), to determine the
    # largest such hypercube that contains only states with max value
    possible_idx_vals = list(range(num_bins_per_dim))
    all_idx_combos = list(itertools.product(possible_idx_vals,
                                            repeat=len(env.obs_space)))
    done = False
    while not done:
        idx_combos_within_bounds = []
        for idx_combo in all_idx_combos:
            within_bounds = True
            for idx_for_dim in idx_combo:
                within_bounds_this_dim = (idx_left <= idx_for_dim <= idx_right)
                within_bounds = (within_bounds and within_bounds_this_dim)
            if within_bounds:
                idx_combos_within_bounds.append(idx_combo)

        print(f"{idx_left}, {idx_right}: {len(idx_combos_within_bounds)} "
              f"within bounds")

        all_max_val = np.all([v[idx_combo] == MAX_VAL for idx_combo in
                             idx_combos_within_bounds])
        done = not all_max_val

        idx_left -= 1
        idx_right += 1
        assert idx_left_min <= idx_left
        assert idx_right_max >= idx_right


if __name__ == "__main__":
    args = parse_args()
    main(args)
