import math
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from rlenvs.mountain_car import make_mountain_car_a_env as make_mc_a
from rlenvs.mountain_car import make_mountain_car_b_env as make_mc_b
from mc_util import POS_MIN, POS_MAX, VEL_MIN, VEL_MAX, ACTION_SET
from pi_npy_policy import PiNpyPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-npy", required=True)
    parser.add_argument("--mc-env-variant", choices=["a", "b", "c"],
                        required=True)
    parser.add_argument("--num-hex-bins", type=int, required=True)
    return parser.parse_args()


def main(args):
    pi_npy = np.load(args.pi_npy)
    env = _make_mc_env(args.mc_env_variant)
    policy = PiNpyPolicy(pi_npy, env)

    (perf, trajs) = env.assess_perf_and_get_trajs(policy)
    for traj in trajs:
        print(len(traj))
    print(f"min traj len: {min([len(traj) for traj in trajs])}")
    print(f"max traj len: {max([len(traj) for traj in trajs])}")
    print(perf)
    print(np.histogram([len(traj) for traj in trajs], bins=15))

    x_min = POS_MIN
    x_max = POS_MAX
    y_min = VEL_MIN
    y_max = VEL_MAX
    aspect = abs((x_max-x_min)/(y_max-y_min))

    xss = []
    yss = []
    for traj in trajs:
        xs = [s[0] for (s, a) in traj]
        ys = [s[1] for (s, a) in traj]
        xss.extend(xs)
        yss.extend(ys)

    plt.hexbin(xss, yss, extent=[x_min, x_max, y_min,
               y_max], gridsize=args.num_hex_bins, cmap="inferno",
               linewidths=0.2)
    plt.gca().axis([x_min, x_max, y_min, y_max])
    plt.xlabel("$x$")
    plt.ylabel("$\dot{x}$")
    xticks = list(np.arange(-1.2, 0.4+0.2, 0.2))
    xticks.append(0.5)
    xticks = sorted(xticks)
    plt.xticks(xticks, fontsize="x-small")
    yticks = list(np.arange(-0.07, 0.07, 0.02))
    yticks.append(0.0)
    yticks = sorted(yticks)
    plt.yticks(yticks, fontsize="x-small")
    plt.colorbar(label="Visit count")
    pi_npy_no_extension = Path(args.pi_npy).with_suffix("")
    plt.savefig(f"{pi_npy_no_extension}_mc_{args.mc_env_variant}_rollout_plot"
                f"_{args.num_hex_bins}_hexbins.png", bbox_inches="tight")
    plt.savefig(f"{pi_npy_no_extension}_mc_{args.mc_env_variant}_rollout_plot"
                f"_{args.num_hex_bins}_hexbins.pdf", bbox_inches="tight")


def _make_mc_env(mc_env_variant):
    if mc_env_variant == "a":
        return make_mc_a()
    elif mc_env_variant == "b":
        return make_mc_b()
    elif mc_env_variant == "c":
        return make_mc_c()
    else:
        raise Exception


if __name__ == "__main__":
    args = parse_args()
    main(args)
