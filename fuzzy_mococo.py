#!/usr/bin/python3
import argparse
import copy
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
from rlenvs.mountain_car import make_mountain_car_a_env as make_mc_a
from rlenvs.mountain_car import make_mountain_car_b_env as make_mc_b
from rlenvs.mountain_car import make_mountain_car_c_env as make_mc_c
from rlenvs.cartpole import make_cartpole_a_env as make_cp_a
from solution import Solution
from zadeh.rule_base import FuzzyRuleBase
from subspecies import (parse_subspecies_tag, validate_subspecies_tags,
                        make_pop_init_pmfs, get_subpop)

from lv_genotype import make_lv_indiv, make_ling_vars
from multi_objective import assign_crowding_dists, assign_pareto_front_ranks
from rb_genotype import make_rb_indiv, make_rule_base
from util import (ACTION_SET, NORMALISE_OBSS, RB_ALLELE_SET,
                  USE_DEFAULT_ACTION_SET, calc_lv_joint_fitness_records,
                  calc_rb_joint_fitness_records, eval_perf,
                  flatten_joint_fitness_mats, make_best_soln_record, make_frbs,
                  make_inference_engine, make_internal_fitness_records)

NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env-name", choices=["mc_a", "mc_b", "mc_c", "cp_a"],
                        required=True)
    parser.add_argument("--subspecies-tags", type=parse_subspecies_tag,
                        nargs="+", required=True)
    parser.add_argument("--ie-and-type", choices=["min", "prod"],
                        required=True)
    parser.add_argument("--ie-or-type", choices=["max", "probor"],
                        required=True)
    parser.add_argument("--ie-agg-type", choices=["bsum", "wavg"],
                        required=True)
    parser.add_argument("--lv-pop-size", type=int, required=True)
    parser.add_argument("--rb-pop-size", type=int, required=True)
    parser.add_argument("--num-gens", type=int, required=True)
    parser.add_argument("--num-collabrs", type=int, required=True)
    parser.add_argument("--lv-tourn-size", type=int, required=True)
    parser.add_argument("--rb-tourn-size", type=int, required=True)
    parser.add_argument("--p-cross-line", type=float, required=True)
    parser.add_argument("--mut-sigma", type=float, required=True)
    parser.add_argument("--p-cross-swap", type=float, required=True)
    parser.add_argument("--p-mut-flip", type=float, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)

    np.random.seed(args.seed)

    env = _make_env(args.env_name)

    subspecies_tags = args.subspecies_tags
    validate_subspecies_tags(subspecies_tags, env)
    (lv_init_pmf, rb_init_pmf) = make_pop_init_pmfs(subspecies_tags)

    inference_engine = make_inference_engine(args.ie_and_type, args.ie_or_type,
                                             args.ie_agg_type, env)

    # p1, p2
    lv_parent_pop = _init_lv_pop(args.lv_pop_size, lv_init_pmf)
    rb_parent_pop = _init_rb_pop(args.rb_pop_size, rb_init_pmf,
                                 inference_engine)
    # q1, q2
    lv_child_pop = []
    rb_child_pop = []

    soln_set = []
    lv_gen_history = {}
    rb_gen_history = {}
    for gen_num in range(args.num_gens + 1):
        lv_pop = lv_parent_pop + lv_child_pop
        rb_pop = rb_parent_pop + rb_child_pop
        _make_lv_phenotypes(lv_pop, env)
        _make_rb_phenotypes(rb_pop, inference_engine)
        if gen_num == 0:
            collabr_map = _select_init_collabrs(lv_parent_pop, rb_parent_pop,
                                                subspecies_ls)
            soln_set = _build_soln_set(lv_parent_pop, rb_parent_pop,
                                       subspecies_ls, collabr_map,
                                       inference_engine)
            _eval_soln_set(soln_set, args.env_perf_seed,
                           args.num_perf_rollouts)
            _assign_indivs_credit(lv_parent_pop, soln_set)
            _assign_indivs_credit(rb_parent_pop, soln_set)
        else:
            _perform_extinction(lv_parent_pop, rb_parent_pop, lv_child_pop,
                                rb_child_pop, subspecies_ls)
            collabr_map = _select_subsq_collabrs(lv_parent_pop, rb_parent_pop,
                                                 subspecies_ls)
            soln_set = _build_soln_set(lv_child_pop, rb_child_pop,
                                       subspecies_ls, collabr_map,
                                       inference_engine)
            _eval_soln_set(soln_set, args.num_perf_rollouts)
            _assign_indivs_credit(lv_child_pop, soln_set)
            _assign_indivs_credit(rb_child_pop, soln_set)

        # do breeding for both pops
        new_lv_pop = _do_breeding(lv_internal_fitness_records,
                                  pop_size=args.lv_pop_size,
                                  num_elites=args.num_lv_elites,
                                  tourn_size=args.lv_tourn_size,
                                  cross_callback={
                                      "func": _line_recombination,
                                      "kwargs": {
                                          "p_cross_line": args.p_cross_line
                                      }
                                  },
                                  mut_callback={
                                      "func": _gaussian_mutation,
                                      "kwargs": {
                                          "sigma": args.mut_sigma
                                      }
                                  })
        new_rb_pop = _do_breeding(rb_internal_fitness_records,
                                  pop_size=args.rb_pop_size,
                                  num_elites=args.num_rb_elites,
                                  tourn_size=args.rb_tourn_size,
                                  cross_callback={
                                      "func": _uniform_crossover,
                                      "kwargs": {
                                          "p_cross_swap": args.p_cross_swap
                                      }
                                  },
                                  mut_callback={
                                      "func": _flip_mutation,
                                      "kwargs": {
                                          "p_mut_flip": args.p_mut_flip
                                      }
                                  })
        lv_pop = new_lv_pop
        rb_pop = new_rb_pop

    _save_data(save_path, lv_gen_history, rb_gen_history, best_soln_record,
               args)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.INFO)


def _make_env(env_name):
    if env_name == "mc_a":
        return make_mc_a()
    elif env_name == "mc_b":
        return make_mc_b()
    elif env_name == "mc_c":
        return make_mc_c()
    elif env_name == "cp_a":
        return make_cp_a()
    else:
        raise Exception


def _init_lv_pop(lv_pop_size, lv_init_pmf):
    subspecies_tag_sample = _sample_subspecies_tags(lv_pop_size,
                                                    lv_init_pmf)
    lv_pop = []
    for subspecies_tag in subspecies_tag_sample:
        lv_pop.append(make_lv_indiv(subspecies_tag))
    return lv_pop


def _init_rb_pop(rb_pop_size, rb_init_pmf, inference_engine):
    subspecies_tag_sample = _sample_subspecies_tags(rb_pop_size,
                                                    rb_init_pmf)
    rb_pop = []
    for subspecies_tag in subspecies_tag_sample:
        rb_pop.append(make_rb_indiv(subspecies_tag, inference_engine))
    return rb_pop


def _sample_subspecies_tags(pop_size, subspecies_init_pmf):
    return np.random.choice(a=list(subspecies_init_pmf.keys()),
                            size=pop_size,
                            p=list(subspecies_init_pmf.values()))


def _make_lv_phenotypes(lv_pop, env):
    for indiv in lv_pop:
        indiv.phenotype = make_ling_vars(indiv.subspecies_tag, indiv.genotype,
                                         env)


def _make_rb_phenotypes(rb_pop, inference_engine):
    for indiv in rb_pop:
        indiv.phenotype = make_rule_base(indiv.subspecies_tag, indiv.genotype,
                                         inference_engine)


def _perform_extinction(lv_parent_pop, rb_parent_pop, lv_child_pop,
                        rb_child_pop, subspecies_ls):
    assert len(lv_child_pop) > 0
    assert len(rb_child_pop) > 0

    p1 = lv_parent_pop
    p2 = rb_parent_pop
    q1 = lv_child_pop
    q2 = rb_child_pop
    for subspecies_tag in subspecies_ls:
        q1_sigma = get_subpop(q1, subspecies_tag)
        p2_sigma = get_subpop(p2, subspecies_tag)
        if len(q1_sigma) == 0 or len(p2_sigma) == 0:
            for indiv in q1_sigma:
                q1.remove(indiv)
            for indiv in p2_sigma:
                p2.remove(indiv)

        q2_sigma = get_subpop(q2, subspecies_tag)
        p1_sigma = get_subpop(p1, subspecies_tag)
        if len(q2_sigma) == 0 or len(p1_sigma) == 0:
            for indiv in q2_sigma:
                q2.remove(indiv)
            for indiv in p1_sigma:
                p1.remove(indiv)

    r1 = p1 + q1
    r2 = p2 + q2
    subspecies_ls_copy = copy.deepcopy(subspecies_ls)
    for subspecies_tag in subspecies_ls_copy:
        # check if there are *any* indivs with given tag in either pop
        # if not, subspecies is extinct
        r1_sigma = get_subpop(r1, subspecies_tag)
        r2_sigma = get_subpop(r2, subspecies_tag)
        if len(r1_sigma) == 0 and len(r2_sigma) == 0:
            subspecies_ls.remove(subspecies_tag)


def _select_init_collabrs(lv_parent_pop, rb_parent_pop, subspecies_ls):
    return _select_collabrs(lv_parent_pop,
                            rb_parent_pop,
                            subspecies_ls,
                            select_func=_single_random_collabr)


def _select_subsq_collabrs(lv_parent_pop, rb_parent_pop, subspecies_ls):
    return _select_collabrs(lv_parent_pop,
                            rb_parent_pop,
                            subspecies_ls,
                            select_func=_best_and_random_collabr)


def _select_collabrs(lv_parent_pop, rb_parent_pop, subspecies_ls, select_func):
    # keys of collabr map are (pop_num, subspecies_tag) tuples: i.e. subpop
    # specifications, values are lists of collabrs (Indiv objs) in subpops
    collabr_map = {}
    pop_num_mapping = {1: lv_parent_pop, 2: rb_parent_pop}
    for (pop_num, pop) in pop_num_mapping.items():
        for subspecies_tag in subspecies_ls:
            subpop = get_subpop(pop, subspecies_tag)
            subpop_collabrs = select_func(subpop)
            collabr_map[(pop_num, subspecies_tag)] = subpop_collabrs
    return collabr_map


def _single_random_collabr(subpop):
    return [np.random.choice(subpop, size=1)]


def _best_and_random_collabr(subpop):
    best_indiv = _get_best_indiv(subpop)
    subpop_no_best = copy.deepcopy(subpop)
    subpop_no_best.remove(best_indiv)
    # also select random indiv if possible
    if len(subpop_no_best) > 0:
        random_indiv = np.random.choice(subpop_no_best)
        return [best_indiv, random_indiv]
    else:
        return [best_indiv]


def _get_best_indiv(subpop):
    best_pfr = min([indiv.pareto_front_rank for indiv in subpop])
    best_front = [
        indiv for indiv in subpop if indiv.pareto_front_rank == best_pfr
    ]
    best_front_crowd_dist_desc = sorted(best_front,
                                        key=lambda indiv: indiv.crowding_dist,
                                        reverse=True)
    best_indiv = best_front_crowd_dist_desc[0]
    return best_indiv


def _build_soln_set(lv_pop, rb_pop, subspecies_ls, collabr_map,
                    inference_engine):
    soln_set = []
    pop_num_mapping = {1: lv_pop, 2: rb_pop}
    for (pop_num, pop) in pop_num_mapping.items():
        for subspecies_tag in subspecies_ls:
            subpop = get_subpop(pop, subspecies_tag)
            if pop_num == 1:
                collabr_pop_num = 2
            else:
                collabr_pop_num = 1
            collabrs = collabr_map[(collabr_pop_num, subspecies_tag)]
            for indiv in subpop:
                for collabr in collabrs:
                    soln_set.append(
                        _make_soln(indiv, collabr, inference_engine))
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
        raise Exception
    frbs = make_frbs(inference_engine,
                     ling_vars=lv_indiv.phenotype,
                     rule_base=rb_indiv.phenotype)
    return Solution(lv_indiv, rb_indiv, frbs)


def _eval_soln_set(soln_set, env, env_perf_seed, num_perf_rollouts):
    # farm out performance evaluation (parallelise over solns)
    with Pool(NUM_CPUS) as pool:
        perfs = pool.starmap(eval_perf,
                             [(env, soln) for soln in soln_set])
    # do complexity evaluation in serial since not expensive
    complexities = [_eval_complexity(soln) for soln in soln_set]

    # assign perfs and complexities to solns
    for (perf, complexity, soln) in zip(perfs, complexities, soln_set):
        # TODO add assertions to check bounds
        soln.perf = perf
        soln.complexity = complexity

    assign_pareto_front_ranks(soln_set)
    assign_crowding_dists(soln_set)


def _eval_complexity(soln):
    # TODO fix
    return soln.frbs.calc_complexity()


def _assign_indivs_credit(pop, soln_set):
    for indiv in pop:
        solns_contained_in = [
            soln for soln in soln_set if soln.does_contain(indiv)
        ]
        pareto_front_ranks = [
            soln.pareto_front_rank for soln in solns_contained_in
        ]
        crowding_dists = [soln.crowding_dist for soln in solns_contained_in]
        # assign best case of each to indiv
        indiv.pareto_front_rank = min(pareto_front_ranks)
        indiv.crowding_dist = max(crowding_dists)


def _do_breeding(internal_fitness_records, pop_size, num_elites, tourn_size,
                 cross_callback, mut_callback):
    assert pop_size % 2 == 0
    assert 0 <= num_elites <= pop_size
    assert (pop_size - num_elites) % 2 == 0

    new_pop = []
    elites = _select_elites(internal_fitness_records, num_elites)
    for elite in elites:
        new_pop.append(elite)
    for _ in range(int((pop_size - num_elites) / 2)):
        parent_a = _selection(internal_fitness_records, tourn_size)
        parent_b = _selection(internal_fitness_records, tourn_size)
        children = _crossover_and_mutate(parent_a, parent_b, cross_callback,
                                         mut_callback)
        for child in children:
            new_pop.append(child)
    assert len(new_pop) == pop_size
    return new_pop


def _select_elites(internal_fitness_records, num_elites):
    fitness_desc = sorted(internal_fitness_records,
                          key=lambda ifr: ifr.fitness,
                          reverse=True)
    elites = []
    for idx in range(num_elites):
        elites.append(fitness_desc[idx].genotype)
    return elites


def _selection(internal_fitness_records, tourn_size):
    """Tournament selection"""
    def _select_random(internal_fitness_records):
        idx = np.random.choice(list(range(0, len(internal_fitness_records))))
        return internal_fitness_records[idx]

    best = _select_random(internal_fitness_records)
    for _ in range(2, (tourn_size + 1)):
        next_ = _select_random(internal_fitness_records)
        if next_.fitness > best.fitness:
            best = next_
    return best.genotype


def _crossover_and_mutate(parent_a, parent_b, cross_callback, mut_callback):
    child_a = copy.deepcopy(parent_a)
    child_b = copy.deepcopy(parent_b)
    _crossover_children(child_a, child_b, cross_callback)
    _mutate_child(child_a, mut_callback)
    _mutate_child(child_b, mut_callback)
    return (child_a, child_b)


def _crossover_children(child_a, child_b, cross_callback):
    cross_func = cross_callback["func"]
    func_kwargs = cross_callback["kwargs"]
    cross_func(child_a, child_b, **func_kwargs)


def _uniform_crossover(child_a, child_b, p_cross_swap):
    """Uniform crossover."""
    assert len(child_a) == len(child_b)
    for idx in range(0, len(child_a)):
        should_swap = np.random.rand() < p_cross_swap
        if should_swap:
            child_a[idx], child_b[idx] = child_b[idx], child_a[idx]


def _line_recombination(child_a, child_b, p_cross_line):
    assert len(child_a) == len(child_b)
    should_cross = np.random.rand() < p_cross_line
    if should_cross:
        alpha = np.random.rand()
        beta = np.random.rand()
        for idx in range(0, len(child_a)):
            a_i = child_a[idx]
            b_i = child_b[idx]
            t = alpha * a_i + (1 - alpha) * b_i
            s = beta * b_i + (1 - beta) * a_i
            assert LV_ALLELE_MIN <= t <= LV_ALLELE_MAX
            assert LV_ALLELE_MIN <= s <= LV_ALLELE_MAX
            child_a[idx] = t
            child_b[idx] = s


def _mutate_child(child, mut_callback):
    mut_func = mut_callback["func"]
    func_kwargs = mut_callback["kwargs"]
    mut_func(child, **func_kwargs)


def _flip_mutation(child, p_mut_flip):
    for idx in range(0, len(child)):
        should_flip = np.random.rand() < p_mut_flip
        if should_flip:
            curr_val = child[idx]
            options = {-1: (0, 2), 0: (-1, 2), 2: (-1, 0)}[curr_val]
            new_val = np.random.choice(options)
            child[idx] = new_val


def _gaussian_mutation(child, sigma):
    mu = 0.0
    for idx in range(0, len(child)):
        curr_val = child[idx]
        noise = np.random.normal(loc=mu, scale=sigma)
        new_val = curr_val + noise
        if new_val < LV_ALLELE_MIN or new_val > LV_ALLELE_MAX:
            # mirror the noise
            new_val = curr_val - noise
        assert LV_ALLELE_MIN <= new_val <= LV_ALLELE_MAX
        child[idx] = new_val


def _save_data(save_path, lv_gen_history, rb_gen_history, best_soln_record,
               args):
    with open(save_path / "lv_gen_history.pkl", "wb") as fp:
        pickle.dump(lv_gen_history, fp)
    with open(save_path / "rb_gen_history.pkl", "wb") as fp:
        pickle.dump(rb_gen_history, fp)
    with open(save_path / "best.pkl", "wb") as fp:
        pickle.dump(best_soln_record, fp)
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
    logging.info(f"Runtime: {elpased:.3f}s with {NUM_CPUS} cpus")
