#!/usr/bin/python3
import argparse
import glob
import logging
import os
import pickle
import shutil
import subprocess
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
from rlenvs.cartpole import make_cartpole_a_env as make_cp_a
from rlenvs.cartpole import make_cartpole_b_env as make_cp_b
from rlenvs.mountain_car import make_mountain_car_a_env as make_mc_a
from rlenvs.mountain_car import make_mountain_car_b_env as make_mc_b
from zadeh.rule_base import FuzzyRuleBase

from ga import run_lv_ga, run_rb_ga
from lv_genotype import make_ling_vars, make_lv_indiv
from mo_constants import MIN_COMPLEXITY_LB
from multi_objective import (ComplexityBounds, assign_crowding_dists,
                             assign_pareto_front_ranks, calc_max_complexity,
                             calc_soln_complexity, calc_soln_perf,
                             crowded_comparison_sort, select_parent_pop)
from rb_genotype import make_rb_indiv, make_rule_base
from soln import Solution
from subspecies import (get_subpop, make_subspecies_pmfs_both_pops,
                        parse_subspecies_tags, sample_subspecies_tags,
                        validate_subspecies_tags)
from util import PopRecord, make_frbs, make_inference_engine

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--env-name",
        choices=["mc_a", "mc_b", "cp_a", "cp_b"],
        required=True)
    parser.add_argument("--subspecies-tags",
                        type=parse_subspecies_tags,
                        required=True)
    parser.add_argument("--subspecies-pmf-base", type=float, required=True)
    parser.add_argument("--ie-and-type",
                        choices=["min", "prod"],
                        required=True)
    parser.add_argument("--ie-or-type",
                        choices=["max", "probor"],
                        required=True)
    parser.add_argument("--ie-agg-type",
                        choices=["bsum", "wavg"],
                        required=True)
    parser.add_argument("--min-complexity", type=int, required=True)
    parser.add_argument("--lv-pop-size", type=int, required=True)
    parser.add_argument("--rb-pop-size", type=int, required=True)
    parser.add_argument("--rb-p-unspec-init", type=float, required=True)
    parser.add_argument("--num-gens", type=int, required=True)
    parser.add_argument("--num-collabrs", type=int, required=True)
    parser.add_argument("--tourn-size", type=int, required=True)
    parser.add_argument("--lv-p-cross-line", type=float, required=True)
    parser.add_argument("--lv-mut-sigma", type=float, required=True)
    parser.add_argument("--rb-p-cross-swap", type=float, required=True)
    parser.add_argument("--rb-p-mut-flip", type=float, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    np.random.seed(args.seed)

    env = _make_env(args.env_name)

    subspecies_tags = args.subspecies_tags
    validate_subspecies_tags(subspecies_tags, env)
    (lv_subspecies_pmf, rb_subspecies_pmf) = \
        make_subspecies_pmfs_both_pops(subspecies_tags,
                                       args.subspecies_pmf_base)
    logging.info(f"lv subspecies pmf: {lv_subspecies_pmf}, "
                 f"rb subspecies pmf: {rb_subspecies_pmf}")

    inference_engine = make_inference_engine(args.ie_and_type, args.ie_or_type,
                                             args.ie_agg_type, env)
    max_complexity = calc_max_complexity(subspecies_tags)
    assert MIN_COMPLEXITY_LB <= args.min_complexity < max_complexity
    complexity_bounds = ComplexityBounds(min=args.min_complexity,
                                         max=max_complexity)

    # p1, p2
    lv_parent_pop = _init_lv_pop(lv_subspecies_pmf, args.lv_pop_size)
    rb_parent_pop = _init_rb_pop(rb_subspecies_pmf, args.rb_pop_size,
                                 inference_engine, complexity_bounds.min,
                                 args.rb_p_unspec_init)
    # q1, q2
    lv_child_pop = []
    rb_child_pop = []

    lv_pops_history = {}
    rb_pops_history = {}
    soln_set_history = {}
    soln_set = None

    for gen_num in range(args.num_gens + 1):
        logging.info(f"Gen {gen_num}")
        lv_comb_pop = lv_parent_pop + lv_child_pop
        rb_comb_pop = rb_parent_pop + rb_child_pop
        _log_subspecies_dists(lv_parent_pop, rb_parent_pop, lv_child_pop,
                              rb_child_pop)
        _validate_subpops_non_empty(gen_num, lv_parent_pop, rb_parent_pop,
                                    lv_child_pop, rb_child_pop,
                                    subspecies_tags)
        _make_lv_phenotypes(lv_comb_pop, env)
        _make_rb_phenotypes(rb_comb_pop, inference_engine)
        if gen_num == 0:
            collabr_map = _select_init_collabrs(lv_parent_pop, rb_parent_pop,
                                                subspecies_tags,
                                                args.num_collabrs)
            soln_set = _build_soln_set(lv_parent_pop, rb_parent_pop,
                                       subspecies_tags, collabr_map,
                                       inference_engine)
            _eval_soln_set(soln_set, env, complexity_bounds)
            _assign_indivs_credit(lv_parent_pop, soln_set)
            _assign_indivs_credit(rb_parent_pop, soln_set)
        else:
            collabr_map = _select_subsq_collabrs(lv_parent_pop, rb_parent_pop,
                                                 subspecies_tags,
                                                 args.num_collabrs)
            soln_set = _build_soln_set(lv_child_pop, rb_child_pop,
                                       subspecies_tags, collabr_map,
                                       inference_engine)
            _eval_soln_set(soln_set, env, complexity_bounds)
            _assign_indivs_credit(lv_child_pop, soln_set)
            _assign_indivs_credit(rb_child_pop, soln_set)

        _update_soln_set_history(soln_set_history, soln_set, gen_num)

        # do parent selection then breeding for both pops
        lv_comb_pop = lv_parent_pop + lv_child_pop
        rb_comb_pop = rb_parent_pop + rb_child_pop
        _assign_ranks_dists_both_pops(lv_comb_pop, rb_comb_pop, env,
                                      complexity_bounds)
        lv_parent_pop = select_parent_pop(gen_num,
                                          pop=lv_comb_pop,
                                          parent_pop_size=args.lv_pop_size,
                                          subspecies_pmf=lv_subspecies_pmf)
        rb_parent_pop = select_parent_pop(gen_num,
                                          pop=rb_comb_pop,
                                          parent_pop_size=args.rb_pop_size,
                                          subspecies_pmf=rb_subspecies_pmf)
        lv_child_pop = run_lv_ga(lv_parent_pop,
                                 child_pop_size=args.lv_pop_size,
                                 subspecies_pmf=lv_subspecies_pmf,
                                 tourn_size=args.tourn_size,
                                 p_cross_line=args.lv_p_cross_line,
                                 mut_sigma=args.lv_mut_sigma)
        rb_child_pop = run_rb_ga(rb_parent_pop,
                                 child_pop_size=args.rb_pop_size,
                                 subspecies_pmf=rb_subspecies_pmf,
                                 tourn_size=args.tourn_size,
                                 p_cross_swap=args.rb_p_cross_swap,
                                 p_mut_flip=args.rb_p_mut_flip,
                                 inference_engine=inference_engine,
                                 min_complexity=complexity_bounds.min)

        _update_pops_history(lv_pops_history, lv_parent_pop, lv_child_pop,
                             gen_num)
        _update_pops_history(rb_pops_history, rb_parent_pop, rb_child_pop,
                             gen_num)
        _log_best_soln_set_rank(soln_set)

    _save_data(save_path, lv_pops_history, rb_pops_history, soln_set_history,
               args)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)


def _make_env(env_name):
    if env_name == "mc_a":
        return make_mc_a()
    elif env_name == "mc_b":
        return make_mc_b()
    elif env_name == "cp_a":
        return make_cp_a()
    elif env_name == "cp_b":
        return make_cp_b()
    else:
        assert False


def _init_lv_pop(lv_subspecies_pmf, lv_pop_size):
    logging.info("Initing lv pop")
    subspecies_tag_sample = sample_subspecies_tags(lv_subspecies_pmf,
                                                   sample_size=lv_pop_size)
    lv_pop = []
    for subspecies_tag in subspecies_tag_sample:
        lv_pop.append(make_lv_indiv(subspecies_tag))
    return lv_pop


def _init_rb_pop(rb_subspecies_pmf, rb_pop_size, inference_engine,
                 min_complexity, rb_p_unspec_init):
    logging.info("Initing rb pop")
    subspecies_tag_sample = sample_subspecies_tags(rb_subspecies_pmf,
                                                   sample_size=rb_pop_size)
    rb_pop = []
    for subspecies_tag in subspecies_tag_sample:
        rb_pop.append(
            make_rb_indiv(subspecies_tag,
                          inference_engine,
                          min_complexity,
                          p_unspec_init=rb_p_unspec_init))
    return rb_pop


def _log_subspecies_dists(lv_parent_pop, rb_parent_pop, lv_child_pop,
                          rb_child_pop):
    lv_parent_dist = _get_subspecies_dist_for_pop(lv_parent_pop)
    logging.debug(f"lv parent subspecies dist: {lv_parent_dist}")
    rb_parent_dist = _get_subspecies_dist_for_pop(rb_parent_pop)
    logging.debug(f"rb parent subspecies dist: {rb_parent_dist}")
    lv_child_dist = _get_subspecies_dist_for_pop(lv_child_pop)
    logging.debug(f"lv child subspecies dist: {lv_child_dist}")
    rb_child_dist = _get_subspecies_dist_for_pop(rb_child_pop)
    logging.debug(f"rb child subspecies dist: {rb_child_dist}")


def _get_subspecies_dist_for_pop(pop):
    num_indivs = len(pop)
    indiv_tags = [indiv.subspecies_tag for indiv in pop]
    unique_indiv_tags = list(set(indiv_tags))
    tags_dist = {}
    for tag in unique_indiv_tags:
        count = len([indiv for indiv in pop if indiv.subspecies_tag == tag])
        frac = count / num_indivs
        tags_dist[tag] = (count, frac)
    return tags_dist


def _validate_subpops_non_empty(gen_num, lv_parent_pop, rb_parent_pop,
                                lv_child_pop, rb_child_pop, subspecies_tags):
    if gen_num == 0:
        pops_to_check = (lv_parent_pop, rb_parent_pop)
        assert len(lv_child_pop) == 0
        assert len(rb_child_pop) == 0
    else:
        pops_to_check = (lv_parent_pop, rb_parent_pop, lv_child_pop,
                         rb_child_pop)
    for pop in pops_to_check:
        for subspecies_tag in subspecies_tags:
            subpop = get_subpop(pop, subspecies_tag)
            assert len(subpop) > 0


def _make_lv_phenotypes(lv_comb_pop, env):
    logging.info("Making lv phenotypes")
    for indiv in lv_comb_pop:
        indiv.phenotype = make_ling_vars(indiv.subspecies_tag, indiv.genotype,
                                         env)


def _make_rb_phenotypes(rb_comb_pop, inference_engine):
    logging.info("Making rb phenotypes")
    for indiv in rb_comb_pop:
        indiv.phenotype = make_rule_base(indiv.subspecies_tag, indiv.genotype,
                                         inference_engine)


def _select_init_collabrs(lv_parent_pop, rb_parent_pop, subspecies_tags,
                          num_collabrs):
    return _select_collabrs(lv_parent_pop,
                            rb_parent_pop,
                            subspecies_tags,
                            num_collabrs,
                            select_func=_random_collabrs_no_replace)


def _select_subsq_collabrs(lv_parent_pop, rb_parent_pop, subspecies_tags,
                           num_collabrs):
    return _select_collabrs(lv_parent_pop,
                            rb_parent_pop,
                            subspecies_tags,
                            num_collabrs,
                            select_func=_balanced_collabrs)


def _select_collabrs(lv_parent_pop, rb_parent_pop, subspecies_tags,
                     num_collabrs, select_func):
    logging.info("Selecting collabrs")
    # keys of collabr map are (pop_num, subspecies_tag) tuples: i.e. subpop
    # specifications, values are lists of collabrs (Indiv objs) in subpops
    collabr_map = {}
    pop_num_mapping = {1: lv_parent_pop, 2: rb_parent_pop}
    for (pop_num, pop) in pop_num_mapping.items():
        for subspecies_tag in subspecies_tags:
            subpop = get_subpop(pop, subspecies_tag)
            subpop_collabrs = \
                _select_collabrs_from_subpop(subpop, num_collabrs, select_func)
            collabr_map[(pop_num, subspecies_tag)] = subpop_collabrs
    return collabr_map


def _select_collabrs_from_subpop(subpop, num_collabrs, select_func):
    assert num_collabrs >= 1
    subpop_size = len(subpop)
    if subpop_size == 0:
        return []
    else:
        num_collabrs = min(subpop_size, num_collabrs)
        return select_func(subpop, num_collabrs)


def _random_collabrs_no_replace(subpop, num_collabrs):
    return list(np.random.choice(subpop, size=num_collabrs, replace=False))


def _balanced_collabrs(subpop, num_collabrs):
    assert num_collabrs <= len(subpop)
    # half exploit, half explore, if odd then give exploit one more
    num_explore = (num_collabrs // 2)
    num_exploit = (num_collabrs - num_explore)

    collabrs = []
    # exploit indivs from sorted subpop
    crowded_comparison_sorted = crowded_comparison_sort(subpop)
    collabrs.extend(crowded_comparison_sorted[0:num_exploit])
    # explore in remaining indivs
    remaining = crowded_comparison_sorted[num_exploit:]
    collabrs.extend(
        list(np.random.choice(remaining, size=num_explore, replace=False)))
    return collabrs


def _build_soln_set(lv_pop, rb_pop, subspecies_tags, collabr_map,
                    inference_engine):
    logging.info("Building soln set")
    soln_set = []
    pop_num_mapping = {1: lv_pop, 2: rb_pop}
    for (pop_num, pop) in pop_num_mapping.items():
        for subspecies_tag in subspecies_tags:
            subpop = get_subpop(pop, subspecies_tag)
            if pop_num == 1:
                collabr_pop_num = 2
            else:
                collabr_pop_num = 1
            # same subspecies tag, other pop
            collabrs = collabr_map[(collabr_pop_num, subspecies_tag)]
            for indiv in subpop:
                for collabr in collabrs:
                    soln_set.append(
                        _make_soln(indiv, collabr, inference_engine))
    logging.info(f"{len(soln_set)} solns built")
    return soln_set


def _make_soln(first_indiv, second_indiv, inference_engine):
    # figure out which indiv is the rule base to get params to frbs factory
    # func in correct order
    if isinstance(first_indiv.phenotype, FuzzyRuleBase):
        rb_indiv = first_indiv
        lv_indiv = second_indiv
    elif isinstance(second_indiv.phenotype, FuzzyRuleBase):
        rb_indiv = second_indiv
        lv_indiv = first_indiv
    else:
        assert False
    frbs = make_frbs(inference_engine,
                     ling_vars=lv_indiv.phenotype,
                     rule_base=rb_indiv.phenotype)
    return Solution(lv_indiv, rb_indiv, frbs)


def _eval_soln_set(soln_set, env, complexity_bounds):
    logging.info("Evaling soln perfs")
    # farm out performance evaluation to multiple processes
    # (parallelise over solns)
    with Pool(_NUM_CPUS) as pool:
        perfs = pool.starmap(calc_soln_perf,
                             [(soln, env) for soln in soln_set])
    # do complexity evaluation in serial since not expensive
    logging.info("Evaling soln complexities")
    complexities = [calc_soln_complexity(soln) for soln in soln_set]

    _assign_perfs_complexities_to_solns(perfs, complexities, soln_set, env,
                                        complexity_bounds)
    assign_pareto_front_ranks(soln_set)
    assign_crowding_dists(soln_set, env, complexity_bounds)


def _assign_perfs_complexities_to_solns(perfs, complexities, soln_set, env,
                                        complexity_bounds):
    for (perf, complexity, soln) in zip(perfs, complexities, soln_set):
        assert env.min_perf <= perf <= env.max_perf
        soln.perf = perf
        assert complexity_bounds.min <= complexity <= complexity_bounds.max
        soln.complexity = complexity


def _assign_indivs_credit(pop, soln_set):
    logging.info("Assigning indivs credit")
    sorted_soln_set = crowded_comparison_sort(soln_set)
    for indiv in pop:
        solns_contained_in = [
            soln for soln in sorted_soln_set if soln.does_contain(indiv)
        ]
        best_soln = solns_contained_in[0]
        indiv.perf = best_soln.perf
        indiv.complexity = best_soln.complexity


def _assign_ranks_dists_both_pops(lv_comb_pop, rb_comb_pop, env,
                                  complexity_bounds):
    for pop in (lv_comb_pop, rb_comb_pop):
        assign_pareto_front_ranks(pop)
        assign_crowding_dists(pop, env, complexity_bounds)


def _update_pops_history(pops_history, parent_pop, child_pop, gen_num):
    pops_history[gen_num] = PopRecord(parents=parent_pop, children=child_pop)


def _update_soln_set_history(soln_set_history, soln_set, gen_num):
    soln_set_history[gen_num] = soln_set


def _log_best_soln_set_rank(soln_set):
    logging.debug("Best rank in soln set")
    best_rank = [soln for soln in soln_set if soln.pareto_front_rank == 1]
    for soln in best_rank:
        logging.debug(f"{soln.subspecies_tag}: {soln.perf}, {soln.complexity}")


def _save_data(save_path, lv_pops_history, rb_pops_history, soln_set_history,
               args):
    with open(save_path / "lv_pops_history.pkl", "wb") as fp:
        pickle.dump(lv_pops_history, fp)
    with open(save_path / "rb_pops_history.pkl", "wb") as fp:
        pickle.dump(rb_pops_history, fp)
    with open(save_path / "soln_set_history.pkl", "wb") as fp:
        pickle.dump(soln_set_history, fp)
    with open(save_path / "var_args.txt", "w") as fp:
        fp.write(str(args))
    _save_python_env_info(save_path)
    _save_py_files(save_path)


def _save_python_env_info(save_path):
    result = subprocess.run(["pip3", "freeze"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return_val = result.stdout.decode("utf-8")
    with open(save_path / "python_env_info.txt", "w") as fp:
        fp.write(str(return_val))


def _save_py_files(save_path):
    for py_file in glob.glob("*.py"):
        shutil.copyfile(Path(py_file), save_path / py_file)


if __name__ == "__main__":
    set_start_method("spawn")  # uses less memory
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")
