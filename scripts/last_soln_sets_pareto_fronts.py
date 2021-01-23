import argparse
import glob
import pickle
import sys

"""Script to extract both:
    1. last soln sets
    2. last pareto fronts
of a group of experiment dirs, and save them into pickle files."""

sys.path.append("/home/Staff/uqjbish3/fuzzy-mococo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-basedir", required=True)
    parser.add_argument("--last-gen-num", type=int, required=True)
    return parser.parse_args()


def main(args):
    experiment_dirs = glob.glob(f"{args.experiments_basedir}/62*")

    last_soln_sets = {}
    for experiment_dir in experiment_dirs:
        with open(f"{experiment_dir}/soln_set_history.pkl", "rb") as fp:
            soln_set_history = pickle.load(fp)
        last_soln_set = soln_set_history[args.last_gen_num]
        experiment_num = experiment_dir.split("/")[1]
        last_soln_sets[experiment_num] = last_soln_set

    last_pareto_fronts = {}
    for (experiment_num, last_soln_set) in last_soln_sets.items():
        last_pareto_front = [soln for soln in last_soln_set if
                             soln.pareto_front_rank == 1]
        last_pareto_fronts[experiment_num] = last_pareto_front

    with open(f"{args.experiments_basedir}/last_soln_sets.pkl", "wb") as fp:
        pickle.dump(last_soln_sets, fp)
    with open(f"{args.experiments_basedir}/last_pareto_fronts.pkl",
              "wb") as fp:
        pickle.dump(last_pareto_fronts, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
