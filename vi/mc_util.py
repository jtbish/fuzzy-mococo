from rlenvs.mountain_car import make_mountain_car_a_env as make_mc

_ENV = make_mc()

pos_dim = _ENV.obs_space[0]
vel_dim = _ENV.obs_space[1]

POS_MIN = pos_dim.lower
POS_MAX = pos_dim.upper
VEL_MIN = vel_dim.lower
VEL_MAX = vel_dim.upper

ACTION_SET = _ENV.action_set
