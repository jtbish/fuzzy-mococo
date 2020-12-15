import argparse
import numpy as np
import itertools

"""Convert Q vals array to policy. Only works for binary action sets.
Values in policy are *idxs into appropriate env action set, unless val is -1 in
which case there is a tie*."""

FIRST_ACTION = 0
SECOND_ACTION = 1
TIE = -1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-npy", required=True)
    return parser.parse_args()


def main(args):
    q = np.load(args.q_npy)
    q_shape = q.shape
    pi_shape = q_shape[:-1]
    equal_num_bins_on_each_dim = (len(set(pi_shape)) == 1)
    assert equal_num_bins_on_each_dim
    num_bins_per_dim = pi_shape[0]
    num_dims = len(pi_shape)
    num_actions = q_shape[-1]
    assert num_actions == 2
    policy = np.empty(shape=pi_shape)

    possible_bin_idxs = list(range(0, num_bins_per_dim))
    all_idx_combos = list(itertools.product(possible_bin_idxs,
                          repeat=num_dims))
    for idx_combo in all_idx_combos:
        q_vals = q[idx_combo]
        assert len(q_vals) == num_actions
        first_q_val = q_vals[0]
        second_q_val = q_vals[1]
        if first_q_val > second_q_val:
            action = FIRST_ACTION
        elif second_q_val > first_q_val:
            action = SECOND_ACTION
        elif first_q_val == second_q_val:
            action = TIE
        else:
            raise Exception
        policy[idx_combo] = action

    np.save(args.q_npy.replace("q", "pi"), policy)


if __name__ == "__main__":
    args = parse_args()
    main(args)
