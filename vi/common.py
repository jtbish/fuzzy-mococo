import math


def calc_dims_step_sizes(obs_space, num_bins_per_dim):
    dims_step_sizes = []
    for dim in obs_space:
        step_size = (dim.upper - dim.lower) / num_bins_per_dim
        dims_step_sizes.append(step_size)
    return dims_step_sizes


def discretise_state(real_state, dims_step_sizes, num_bins_per_dim, env):
    max_bin_idx = (num_bins_per_dim - 1)
    dim_bin_idxs = []
    for (real_state_component, dim, step_size) in \
            zip(real_state, env.obs_space, dims_step_sizes):
        dim_bin_idx = math.floor(
            (real_state_component - dim.lower) / step_size)
        # take care of edge case where component of real state is equal to
        # maximum value of its dim, hence when doing floor the bin idx is one
        # over the maximum.
        dim_bin_idx = min(dim_bin_idx, max_bin_idx)
        dim_bin_idxs.append(dim_bin_idx)
    return tuple(dim_bin_idxs)
