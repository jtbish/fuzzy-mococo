import argparse
import pickle
import sys

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append("/home/Staff/uqjbish3/fuzzy-mococo")

PERF_TICKS = {
    "mc": [-90 - x for x in range(0, (110 + 10), 10)],
    "cp": [10 + x for x in range(0, (190 + 10), 10)]
}
GRIDLINE_ALPHA = 0.33
NUM_SOLNS_TICKS = [10**0, 10**1, 10**2, 10**3, 10**4]
MC_HUE_ORDER = [(2, 2), (3, 3), (4, 4), (5, 5)]
CP_HUE_ORDER = [(2, 2, 2, 2), (3, 2, 3, 2)]
STRIP_ALPHA = 0.125
STRIP_JITTER = 0.125
STRIP_SIZE = 6
ZOOM_COLOR = "magenta"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-basedir", required=True)
    return parser.parse_args()


def main(args):
    with open(f"{args.experiments_basedir}/last_pareto_fronts.pkl",
              "rb") as fp:
        lpfs = pickle.load(fp)

    lpfs_flat = []
    for lpf in lpfs.values():
        lpfs_flat.extend(lpf)

    comps = [soln.complexity for soln in lpfs_flat]
    perfs = [soln.perf for soln in lpfs_flat]
    print(min(perfs), max(perfs))
    ss_tags = [soln.subspecies_tag for soln in lpfs_flat]

    widths = [1]
    heights = [1, 3, 1.5]  # height ratios
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots(nrows=3,
                            ncols=1,
                            figsize=(8, 7),
                            gridspec_kw=gs_kw)

    # do the stripplot first so that the x axis coords can be initialised in
    # range(0, num_unique_comps), then do the matplotlib barchart with these x
    # coords
    if args.experiments_basedir == "mc":
        hue_order = MC_HUE_ORDER
    elif args.experiments_basedir == "cp":
        hue_order = CP_HUE_ORDER
    else:
        assert False
    g2 = sns.stripplot(x=comps,
                       y=perfs,
                       hue=ss_tags,
                       hue_order=hue_order,
                       alpha=STRIP_ALPHA,
                       jitter=STRIP_JITTER,
                       size=STRIP_SIZE,
                       ax=axs[1])
    g2.set_ylabel("Performance")
    if args.experiments_basedir in PERF_TICKS.keys():
        g2.set_yticks(PERF_TICKS[args.experiments_basedir])
    g2.legend(loc="lower right", title="Subspecies tags")
    axs[1].grid(axis="both", alpha=GRIDLINE_ALPHA)

    (comps, counts) = np.unique(comps, return_counts=True)
    print(list(zip(comps, counts)))
    axs[0].bar(x=list(range(len(counts))),
               height=counts,
               color="grey",
               width=0.5)
    axs[0].set_yscale("log")
    axs[0].set_xticks(range(0, len(comps)))
    axs[0].set_xticklabels([str(comp) for comp in comps])
    axs[0].set_yticks(NUM_SOLNS_TICKS)
    axs[0].set_ylim([0, NUM_SOLNS_TICKS[-1]])
    axs[0].set_ylabel("Num solutions")
    axs[0].grid(axis="y", alpha=GRIDLINE_ALPHA)

    _draw_zoom_box(args.experiments_basedir, ax=axs[1])
    _plot_zoom_area(args.experiments_basedir, lpfs_flat, ax=axs[2])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.175)
    plt.savefig(f"{args.experiments_basedir}/last_pareto_fronts_plot.pdf",
                bbox_inches="tight")


def _draw_zoom_box(experiments_basedir, ax):
    if experiments_basedir == "mc":
        rect = patches.Rectangle((2.667, -98),
                                 9.667,
                                 -4,
                                 linewidth=1,
                                 edgecolor=ZOOM_COLOR,
                                 facecolor="none")
    elif experiments_basedir == "cp":
        rect = patches.Rectangle((6.667, 200),
                                 4.667,
                                 -5,
                                 linewidth=1,
                                 edgecolor=ZOOM_COLOR,
                                 facecolor="none")
    else:
        assert False
    ax.add_patch(rect)


def _plot_zoom_area(experiments_basedir, lpfs_flat, ax):
    if experiments_basedir == "mc":
        # area is ([5, 16], [-102, -98])
        def _in_area(soln):
            return (5 <= soln.complexity <= 16) and (-102 <= soln.perf <= -98)
        solns_filtered = [soln for soln in lpfs_flat if _in_area(soln)]
        comps = [soln.complexity for soln in solns_filtered]
        perfs = [soln.perf for soln in solns_filtered]
        ss_tags = [soln.subspecies_tag for soln in solns_filtered]

        hue_order = MC_HUE_ORDER
        g3 = sns.stripplot(x=comps,
                           y=perfs,
                           hue=ss_tags,
                           hue_order=hue_order,
                           alpha=STRIP_ALPHA,
                           jitter=STRIP_JITTER,
                           size=STRIP_SIZE,
                           ax=ax)
        g3.set_xlabel("Complexity")
        g3.set_ylabel("Performance")
        g3.set_yticks(np.arange(-102, -98 + 0.5, 0.5))
    elif experiments_basedir == "cp":
        # area is ([9, 13], [195, 200])
        def _in_area(soln):
            return (9 <= soln.complexity <= 13) and (195 <= soln.perf <= 200)
        solns_filtered = [soln for soln in lpfs_flat if _in_area(soln)]
        comps = [soln.complexity for soln in solns_filtered]
        perfs = [soln.perf for soln in solns_filtered]
        ss_tags = [soln.subspecies_tag for soln in solns_filtered]

        hue_order = CP_HUE_ORDER
        g3 = sns.stripplot(x=comps,
                           y=perfs,
                           hue=ss_tags,
                           hue_order=hue_order,
                           alpha=STRIP_ALPHA,
                           jitter=STRIP_JITTER,
                           size=STRIP_SIZE,
                           ax=ax)
        g3.set_xlabel("Complexity")
        g3.set_ylabel("Performance")
        g3.set_yticks(np.arange(195, 200 + 0.5, 0.5))
    else:
        assert False

    ax.grid(axis="both", alpha=GRIDLINE_ALPHA)
    ax.get_legend().remove()
    ax.spines['bottom'].set_color(ZOOM_COLOR)
    ax.spines['top'].set_color(ZOOM_COLOR)
    ax.spines['right'].set_color(ZOOM_COLOR)
    ax.spines['left'].set_color(ZOOM_COLOR)


if __name__ == "__main__":
    args = parse_args()
    main(args)
