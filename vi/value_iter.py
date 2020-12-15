import argparse
import copy
import itertools
import math

import numpy as np

from rlenvs.cartpole import make_cartpole_a_env as make_cp
from rlenvs.mountain_car import make_mountain_car_a_env as make_mc
from common import discretise_state, calc_dims_step_sizes

_CONVERGE_TOL = 1e-50
_TERMINAL_DISCRETE_STATE = "DUMMY_TERM"
_TERMINAL_DISCRETE_STATE_VALUE = 0.0
_TRANSITION_PROB = 1.0
_GAMMA = 1.0

_VALUE_BOUNDS = {"mc": {"min": None, "max": 0}, "cp": {"min": 0, "max": 200}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", choices=["mc", "cp"], required=True)
    parser.add_argument("--num-bins-per-dim", type=int, required=True)
    return parser.parse_args()


def main(args):
    env = _make_env(args.env_name)
    num_bins_per_dim = args.num_bins_per_dim
    assert num_bins_per_dim >= 2

    obs_space = env.obs_space
    num_ref_points_per_dim = num_bins_per_dim + 1
    dims_ref_points = _compute_dims_ref_points(obs_space,
                                               num_ref_points_per_dim)
    dims_step_sizes = calc_dims_step_sizes(obs_space, num_bins_per_dim)

    num_dims = len(env.obs_space)
    discrete_states = _calc_discrete_states(num_bins_per_dim, num_dims,
                                            dims_ref_points)
    action_set = env.action_set
    P = _build_transition_matrix(discrete_states, env, dims_step_sizes,
                                 num_bins_per_dim, num_dims)
    v, q = _do_value_iteration(num_bins_per_dim, num_dims, action_set,
                               discrete_states, P, args.env_name)
    _save_arrs(v, q, args.env_name, num_bins_per_dim)


def _make_env(env_name):
    if env_name == "mc":
        return make_mc()
    elif env_name == "cp":
        return make_cp()
    else:
        raise Exception


def _compute_dims_ref_points(obs_space, num_ref_points_per_dim):
    dims_ref_points = []
    for dim in obs_space:
        dims_ref_points.append(
            np.linspace(dim.lower,
                        dim.upper,
                        num=num_ref_points_per_dim,
                        endpoint=True))
    return dims_ref_points



def _calc_discrete_states(num_bins_per_dim, num_dims, dims_ref_points):
    # each discrete state is idenified by n-tuple giving idxs of bins
    # along dim axes, along with a "representative" continuous state that is
    # the centroid of the bin: this repr. state is used when querying the
    # actual transition function in continuous space - see func that builds
    # transition matrix
    possible_bin_idxs_for_dim = list(range(0, num_bins_per_dim))
    all_idx_combos = list(
        itertools.product(possible_bin_idxs_for_dim, repeat=num_dims))
    discrete_states = {}
    for idx_combo in all_idx_combos:
        repr_real_state = []
        assert len(idx_combo) == len(dims_ref_points)
        for (idx_on_dim, dim_ref_points) in zip(idx_combo, dims_ref_points):
            # compute the midpoint for this bucket on this dim
            lower_ref_point = dim_ref_points[idx_on_dim]
            upper_ref_point = dim_ref_points[idx_on_dim + 1]
            midpoint = (lower_ref_point + upper_ref_point) / 2
            repr_real_state.append(midpoint)
        repr_real_state = tuple(repr_real_state)
        discrete_states[idx_combo] = repr_real_state
    return discrete_states


def _build_transition_matrix(discrete_states, env, dims_step_sizes,
                             num_bins_per_dim, num_dims):
    P = {}
    discrete_state_count = 0
    for (discrete_state, repr_real_state) in discrete_states.items():
        print(f"Building P state num {discrete_state_count}")
        action_transitions = {}
        for action in env.action_set:
            transitions = []
            primed_env = _prime_env_with_real_state(env, repr_real_state)
            (real_next_state, reward, is_terminal) = primed_env.step(action)
            if not is_terminal:
                # find out what discrete state the real next state corresponds
                # to
                discrete_next_state = \
                    discretise_state(real_next_state, dims_step_sizes,
                                     num_bins_per_dim, env)
            else:
                # terminal transition, use dummy terminal state
                discrete_next_state = _TERMINAL_DISCRETE_STATE
            # (prob, new_state, reward, done)
            transitions.append(
                (_TRANSITION_PROB, discrete_next_state, reward, is_terminal))
            action_transitions[action] = transitions
        P[discrete_state] = action_transitions
        discrete_state_count += 1
    return P


def _prime_env_with_real_state(env, real_state):
    clone = copy.deepcopy(env)
    clone.reset()
    clone._wrapped_env.unwrapped.state = real_state
    return clone


def _do_value_iteration(num_bins_per_dim, num_dims, action_set,
                        discrete_states, P, env_name):
    v_shape = tuple([num_bins_per_dim] * num_dims)
    num_actions = len(action_set)
    q_shape = tuple([num_bins_per_dim] * num_dims + [num_actions])
    v_old = np.zeros(v_shape)
    q = np.zeros(q_shape)
    converged = False
    iter_num = 0
    while not converged:
        print(f"VI iter {iter_num}")
        v_new = np.copy(v_old)
        for discrete_state in discrete_states.keys():
            for action in action_set:
                transitions = P[discrete_state][action]
                expected_reward = sum(
                    [prob * reward for (prob, _, reward, _) in transitions])
                expected_value = 0.0
                for (prob, discrete_next_state, _, _) in transitions:
                    if discrete_next_state != _TERMINAL_DISCRETE_STATE:
                        expected_value += (prob * v_old[discrete_next_state])
                    else:
                        expected_value += \
                            (prob * _TERMINAL_DISCRETE_STATE_VALUE)
                action_idx = list(action_set).index(action)
                q_idx = tuple(list(discrete_state) + [action_idx])
                total_value = expected_reward + _GAMMA * expected_value
                total_value = _bound_value(total_value, env_name)
                q[q_idx] = total_value
            v_new[discrete_state] = np.max(q[discrete_state])
        converged = _has_converged(v_old, v_new)
        v_old = v_new
        iter_num += 1
    return v_new, q


def _bound_value(value, env_name):
    bounds = _VALUE_BOUNDS[env_name]
    min_bound = bounds["min"]
    max_bound = bounds["max"]
    if min_bound is not None:
        value = max(value, min_bound)
    if max_bound is not None:
        value = min(value, max_bound)
    return value


def _save_arrs(v, q, env_name, num_bins_per_dim):
    np.save(f"{env_name}_v_num_bins_{num_bins_per_dim}.npy", v)
    np.save(f"{env_name}_q_num_bins_{num_bins_per_dim}.npy", q)


def _has_converged(v_old, v_new):
    max_diff = np.max(np.abs(v_old - v_new))
    print(f"Max diff in V: {max_diff}")
    if max_diff < _CONVERGE_TOL:
        print("Converged, exiting...")
        return True
    else:
        print("Not converged, running another iter...")
        return False


if __name__ == "__main__":
    args = parse_args()
    main(args)
