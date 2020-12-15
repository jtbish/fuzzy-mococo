import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from mc_util import POS_MIN, POS_MAX, VEL_MIN, VEL_MAX
from policy import FIRST_ACTION, SECOND_ACTION, TIE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-npy", required=True)
    return parser.parse_args()


def main(args):
    pi = np.load(args.pi_npy)
    assert pi.shape[0] == pi.shape[1]
    num_bins = pi.shape[0]

    colors = np.empty(shape=(num_bins, num_bins, 3))  # RGB
    left_color = mcolors.to_rgb(mcolors.CSS4_COLORS['lightcoral'])
    right_color = mcolors.to_rgb(mcolors.CSS4_COLORS['lightgreen'])
    tie_color = mcolors.to_rgb(mcolors.CSS4_COLORS['black'])
    for i in range(0, num_bins):
        for j in range(0, num_bins):
            action_idx = pi[i][j]
            if action_idx == FIRST_ACTION:
                color = left_color
            elif action_idx == SECOND_ACTION:
                color = right_color
            elif action_idx == TIE:
                color = tie_color
            else:
                assert False
            # note idx is [j][i] *not* [i][j], to make pos feature on the x
            # axis, vel feature on y axis since img is specified as
            # rows,columns
            colors[j][i] = np.asarray(color)

    x_min = POS_MIN
    x_max = POS_MAX
    y_min = VEL_MIN
    y_max = VEL_MAX
    aspect = abs((x_max-x_min)/(y_max-y_min))
    plt.imshow(colors, origin="lower", extent=[x_min,
               x_max, y_min, y_max], aspect=aspect, interpolation='none',
               resample=False)
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

    plt.legend(handles=[mpatches.Patch(color=left_color, label="Left"),
               mpatches.Patch(color=right_color, label="Right"),
               mpatches.Patch(color=tie_color, label="Tie")],
               loc="center left", bbox_to_anchor=(1, 0.5))

    # magic dpi number to make actual plot have 1000*1000 pixels
    dpi = 271.35
    pi_npy_no_extension = Path(args.pi_npy).with_suffix("")
    plt.savefig(f"{pi_npy_no_extension}_plot.png",
                bbox_inches="tight", dpi=dpi)
    plt.savefig(f"{pi_npy_no_extension}_plot.pdf",
                bbox_inches="tight", dpi=dpi)


if __name__ == "__main__":
    args = parse_args()
    main(args)
