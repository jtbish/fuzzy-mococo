import argparse
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rlenvs.cartpole import make_cartpole_a_env as make_cp
from rlenvs.mountain_car import make_mountain_car_a_env as make_mc
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


prop_cycle = matplotlib.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

sys.path.append("/home/Staff/uqjbish3/fuzzy-mococo")

MU_TICKS = [0, 0.5, 1]
MU_LIM = [0, 1.05]
MF_LABEL_TEXT_Y = 0.5
TICKS_FONTSIZE = "xx-small"
AXIS_LABEL_FONTSIZE = "small"


def tick_formatter_no_trailing_zeros(x, pos):
    if x.is_integer():
        return str(int(x)) + ".0"
    else:
        return str(x)


formatter = FuncFormatter(tick_formatter_no_trailing_zeros)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-basedir", required=True)
    return parser.parse_args()


def main(args):
    with open(f"{args.experiments_basedir}/repr_best_soln.pkl", "rb") as fp:
        best = pickle.load(fp)
    if "mc" in args.experiments_basedir:
        _mc_plot(best, args.experiments_basedir)
    elif "cp" in args.experiments_basedir:
        _cp_plot(best, args.experiments_basedir)
    else:
        assert False

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.075)
    plt.savefig(f"{args.experiments_basedir}/best_mf_plot.pdf",
                bbox_inches="tight")


def _mc_plot(best, experiments_basedir):
    env = make_mc()
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 7))
    subspecies_tag = best.subspecies_tag

    # plot the curvature of the bowl
    ax = axs[0]
    xs = np.linspace(start=-1.2, stop=0.5, num=1000, endpoint=True)
    ys = [0.45 * np.sin(3 * x) + 0.55 for x in xs]
    ax.plot(xs, ys)
    ax.set_xlim([-1.2, 0.5])
    ax.set_ylabel("$h$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICKS_FONTSIZE)
    ax.text(-0.75, 0.35, r"$h = 0.45 \cdot sin(3x)+0.55$", color=colors[0])

    for idx in range(0, len(env.obs_space)):
        num_mfs = subspecies_tag[idx]
        mfs = (best.lv_indiv.phenotype[idx]).membership_funcs
        assert len(mfs) == num_mfs
        axs_idx = (idx + 1)
        ax = axs[axs_idx]
        for mf_idx in range(0, num_mfs):
            mf = mfs[mf_idx]
            points_to_plot = _remove_trap_boundary_points(
                mf.points, mf_idx, num_mfs)
            _plot_lines(points_to_plot, ax=ax)

        feature_dim = env.obs_space[idx]
        ax.set_xlim([feature_dim.lower, feature_dim.upper])
        ax.set_ylim(MU_LIM)
        ax.set_yticks(MU_TICKS)
        ax.tick_params(axis="both", labelsize=TICKS_FONTSIZE)
        ax.set_ylabel("Degree of membership", fontsize=AXIS_LABEL_FONTSIZE)

    _mc_add_x_ticks(axs)
    _mc_add_x_labels(axs)
    _mc_annotate_mfs(axs)


def _cp_plot(best, experiments_basedir):
    env = make_cp()
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(7, 5))
    subspecies_tag = best.subspecies_tag

    for idx in range(0, len(env.obs_space)):
        num_mfs = subspecies_tag[idx]
        mfs = (best.lv_indiv.phenotype[idx]).membership_funcs
        assert len(mfs) == num_mfs
        axs_idx = ((idx // 2), (idx % 2))
        ax = axs[axs_idx]
        for mf_idx in range(0, num_mfs):
            mf = mfs[mf_idx]
            points_to_plot = _remove_trap_boundary_points(
                mf.points, mf_idx, num_mfs)
            _plot_lines(points_to_plot, ax=ax)

        feature_dim = env.obs_space[idx]
        ax.set_xlim([feature_dim.lower, feature_dim.upper])
        ax.set_ylim(MU_LIM)
        ax.set_yticks(MU_TICKS)
        ax.tick_params(axis="both", labelsize=TICKS_FONTSIZE)

    axs[0, 0].set_ylabel("Degree of membership", fontsize=AXIS_LABEL_FONTSIZE)
    axs[1, 0].set_ylabel("Degree of membership", fontsize=AXIS_LABEL_FONTSIZE)
    _cp_add_x_ticks(axs)
    _cp_add_x_labels(axs)
    _cp_annotate_mfs(axs)


def _plot_lines(points_to_plot, ax):
    xs = [p.x for p in points_to_plot]
    ys = [p.y for p in points_to_plot]
    ax.plot(xs, ys)


def _remove_trap_boundary_points(points_to_plot, mf_idx, num_mfs):
    if mf_idx == 0:
        # remove first point pair
        points_to_plot = points_to_plot[1:]
    elif mf_idx == (num_mfs - 1):
        # remove last point pair
        points_to_plot = points_to_plot[:-1]
    return points_to_plot


def _mc_add_x_ticks(axs):
    xticks_pos = list(np.arange(-1.2, 0.5 + 0.1, 0.1))
    axs[0].set_xticks(xticks_pos)
    axs[1].set_xticks(xticks_pos)

    xticks_vel = list(np.arange(-0.07, 0.07, 0.01))
    axs[2].set_xticks(xticks_vel)


def _mc_add_x_labels(axs):
    axs[0].set_xlabel("$x$", fontsize=AXIS_LABEL_FONTSIZE)
    axs[1].set_xlabel("$x$", fontsize=AXIS_LABEL_FONTSIZE)
    axs[2].set_xlabel(r"$\dot{x}$", fontsize=AXIS_LABEL_FONTSIZE)


def _mc_annotate_mfs(axs):
    axs[1].text(-1.1, MF_LABEL_TEXT_Y, "FL", color=colors[0])
    axs[1].text(-0.675, MF_LABEL_TEXT_Y, "L", color=colors[1])
    axs[1].text(-0.275, MF_LABEL_TEXT_Y, "R", color=colors[2])
    axs[1].text(0.3, MF_LABEL_TEXT_Y, "FR", color=colors[3])

    axs[2].text(-0.06, MF_LABEL_TEXT_Y, "VL", color=colors[0])
    axs[2].text(-0.025, MF_LABEL_TEXT_Y, "L", color=colors[1])
    axs[2].text(0.02, MF_LABEL_TEXT_Y, "H", color=colors[2])
    axs[2].text(0.055, MF_LABEL_TEXT_Y, "VH", color=colors[3])


def _cp_add_x_ticks(axs):
    xticks_cart_pos = list(np.arange(-2.0, 2.0 + 0.5, 0.5))
    xticks_cart_pos.extend([-2.4, 2.4])
    xticks_cart_pos = sorted(xticks_cart_pos)
    axs[0, 0].set_xticks(xticks_cart_pos)
    axs[0, 0].xaxis.set_major_formatter(formatter)

    xticks_cart_vel = list(np.arange(-1.5, 1.5 + 0.5, 0.5))
    xticks_cart_vel.extend([-2.25, 2.25])
    xticks_cart_vel = sorted(xticks_cart_vel)
    axs[0, 1].set_xticks(xticks_cart_vel)
    axs[0, 1].xaxis.set_major_formatter(formatter)

    xticks_pole_ang = [(-12 + x) * np.pi / 180
                       for x in np.arange(0, 24 + 3, 3)]
    labels = []
    for x in np.arange(0, 24 + 3, 3):
        if x == 12:
            label = "0"
        else:
            label = r"$\frac{" + f"{-12+x}" + r"\pi}{180}$"
        labels.append(label)
    axs[1, 0].set_xticks(xticks_pole_ang)
    axs[1, 0].set_xticklabels(labels)

    xticks_pole_vel = list(np.arange(-3.0, 3.0 + 0.75, 0.75))
    xticks_pole_vel.extend([-3.5, 3.5])
    xticks_pole_vel = sorted(xticks_pole_vel)
    axs[1, 1].set_xticks(xticks_pole_vel)
    axs[1, 1].xaxis.set_major_formatter(formatter)


def _cp_add_x_labels(axs):
    axs[0, 0].set_xlabel("$x$", fontsize=AXIS_LABEL_FONTSIZE)
    axs[0, 1].set_xlabel(r"$\dot{x}$", fontsize=AXIS_LABEL_FONTSIZE)
    axs[1, 0].set_xlabel(r"$\theta$", fontsize=AXIS_LABEL_FONTSIZE)
    axs[1, 1].set_xlabel(r"$\dot{\theta}$", fontsize=AXIS_LABEL_FONTSIZE)


def _cp_annotate_mfs(axs):
    axs[0, 0].text(-2.0, MF_LABEL_TEXT_Y, "L", color=colors[0])
    axs[0, 0].text(-0.5, MF_LABEL_TEXT_Y, "M", color=colors[1])
    axs[0, 0].text(1.3, MF_LABEL_TEXT_Y, "H", color=colors[2])

    axs[0, 1].text(-1.5, MF_LABEL_TEXT_Y, "L", color=colors[0])
    axs[0, 1].text(0.7, MF_LABEL_TEXT_Y, "H", color=colors[1])

    axs[1, 0].text((-10 * np.pi) / 180, MF_LABEL_TEXT_Y, "L", color=colors[0])
    axs[1, 0].text((-1.75 * np.pi) / 180,
                   MF_LABEL_TEXT_Y,
                   "M",
                   color=colors[1])
    axs[1, 0].text((8.5 * np.pi) / 180, MF_LABEL_TEXT_Y, "H", color=colors[2])

    axs[1, 1].text(-0.9, MF_LABEL_TEXT_Y, "L", color=colors[0])
    axs[1, 1].text(2.25, MF_LABEL_TEXT_Y, "H", color=colors[1])


if __name__ == "__main__":
    args = parse_args()
    main(args)
