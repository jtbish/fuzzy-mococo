from common import calc_dims_step_sizes, discretise_state
from policy import TIE


class PiNpyPolicy:
    """Class to interpret a discretised policy numpy array for a given
    environment."""
    def __init__(self, pi_npy, env):
        self._pi_npy = pi_npy
        self._env = env
        pi_shape = self._pi_npy.shape
        equal_num_bins_on_each_dim = (len(set(pi_shape)) == 1)
        assert equal_num_bins_on_each_dim
        self._num_bins_per_dim = self._pi_npy.shape[0]
        self._dims_step_sizes = calc_dims_step_sizes(self._env.obs_space,
                                                     self._num_bins_per_dim)
        # use first action in action set as default
        self._default_first_action = self._env.action_set[0]
        self._last_action = None

    def classify(self, real_state):
        discrete_state = discretise_state(real_state, self._dims_step_sizes,
                                          self._num_bins_per_dim, self._env)
        pi_idx_combo = discrete_state

        pi_val = int(self._pi_npy[pi_idx_combo])
        if pi_val != TIE:
            # unambiguous action, use it
            action_set_idx = pi_val
            action = self._env.action_set[action_set_idx]
        else:
            # tie, repeat last action if possible
            if self._last_action is not None:
                action = self._last_action
            else:
                action = self._default_first_action
        self._last_action = action
        return action
