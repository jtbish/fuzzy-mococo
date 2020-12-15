import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mc_util import POS_MIN, POS_MAX, VEL_MIN, VEL_MAX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v-npy", required=True)
    return parser.parse_args()


def main(args):
    v = np.load(args.v_npy)
    # first axis of v is pos, second is vel, want pos on x axis, vel on y axis
    # so transpose s.t. vel is first axis (row of image), pos is second axis
    # (col of image)
    v = np.transpose(v)
    assert v.shape[0] == v.shape[1]

    x_min = POS_MIN
    x_max = POS_MAX
    y_min = VEL_MIN
    y_max = VEL_MAX
    aspect = abs((x_max-x_min)/(y_max-y_min))
    plt.imshow(v, origin="lower", extent=[x_min,
               x_max, y_min, y_max], aspect=aspect, interpolation="none",
               resample=False, cmap="plasma")
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
    cbar_ticks = list(np.arange(0.0, -100+(-10), -10))
    cbar_ticks.append(np.max(v))
    cbar_ticks.append(np.min(v))
    cbar_ticks = sorted(cbar_ticks)
    plt.colorbar(ticks=cbar_ticks, label="Value")

    # magic dpi number to make actual plot have 1000*1000 pixels
    dpi = 271.35
    v_npy_no_extension = Path(args.v_npy).with_suffix("")
    plt.savefig(f"{v_npy_no_extension}_plot.png", bbox_inches="tight", dpi=dpi)
    plt.savefig(f"{v_npy_no_extension}_plot.pdf", bbox_inches="tight", dpi=dpi)


if __name__ == "__main__":
    args = parse_args()
    main(args)
