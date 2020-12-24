import logging
import math

import numpy as np
from zadeh.error import UndefinedMappingError

MIN_DOMINATION_COUNT = 0
MIN_PARETO_FRONT_RANK = 1
MIN_CROWDING_DIST = 0

MIN_COMPLEXITY = 4


def calc_soln_perf(soln, env):
    try:
        return env.assess_perf(soln.frbs)
    except UndefinedMappingError:
        # hole in the frbs, use min perf
        return env.min_perf


def calc_soln_complexity(soln):
    return soln.frbs.calc_complexity()


def calc_max_complexity(subspecies_tags):
    max_possible_fuzzy_decision_regions = \
        max([np.prod(subspecies_tag) for subspecies_tag in subspecies_tags])
    return max_possible_fuzzy_decision_regions


def assign_pareto_front_ranks(soln_set):
    """Second part of fast-non-dominated-sort algorithm from NSGA-II paper,
    except not explicitly storing and returning fronts but rather just storing
    front ranks of each soln."""
    logging.info("Assigning pareto front ranks")
    _find_dominations(soln_set)

    # find first front
    first_front = []
    for soln in soln_set:
        if soln.domination_count == 0:
            soln.pareto_front_rank = 1
            first_front.append(soln)

    # find subsequent fronts
    curr_front_number = 1
    curr_front = first_front
    while len(curr_front) != 0:
        logging.debug(f"Trying to find front number {curr_front_number + 1}")
        next_front = []
        for soln in curr_front:
            for dominated_soln in soln.dominated_set:
                dominated_soln.domination_count -= 1
                if dominated_soln.domination_count == 0:
                    dominated_soln.pareto_front_rank = curr_front_number + 1
                    next_front.append(dominated_soln)
        curr_front_number += 1
        curr_front = next_front
    assert np.all([soln.domination_count == 0 for soln in soln_set])


def _find_dominations(soln_set):
    """First part of fast-non-dominated-sort algorithm from NSGA-II paper"""
    _init_soln_set_domination_attrs(soln_set)
    for soln in soln_set:
        other_solns = [soln_ for soln_ in soln_set if soln_ != soln]
        for other_soln in other_solns:
            if _does_dominate(soln, other_soln):
                soln.dominated_set = soln.dominated_set + [other_soln]
            elif _does_dominate(other_soln, soln):
                soln.domination_count += 1


def _init_soln_set_domination_attrs(soln_set):
    for soln in soln_set:
        soln.dominated_set = []
        soln.domination_count = MIN_DOMINATION_COUNT


def _does_dominate(soln_a, soln_b):
    """Does soln_a Pareto dominate soln_b?
    To dominate soln_b, soln_a must be strictly better in at least one
    objective and no worse in all others.

    Essentials of Metaheuristics Alg. 98: Pareto Domination"""
    res = False
    for objective_func in (_perf_objective, _complexity_objective):
        if _is_better(objective_func, soln_a, soln_b):
            res = True
        elif _is_better(objective_func, soln_b, soln_a):
            return False
    return res


def _perf_objective(soln):
    return soln.perf


def _complexity_objective(soln):
    return soln.complexity


def _is_better(objective_func, first_soln, second_soln):
    """Is first_soln 'better than' second_soln according to objective_func?"""
    if objective_func == _perf_objective:
        # higher is better
        return objective_func(first_soln) > objective_func(second_soln)
    elif objective_func == _complexity_objective:
        # lower is better
        return objective_func(first_soln) < objective_func(second_soln)
    else:
        raise Exception


def assign_crowding_dists(soln_set, env, max_complexity):
    logging.info("Assigning crowding dists")
    soln_pfrs = [soln.pareto_front_rank for soln in soln_set]
    min_pfr = min(soln_pfrs)
    assert min_pfr == 1
    max_pfr = max(soln_pfrs)
    for pfr in range(min_pfr, (max_pfr + 1)):
        pareto_front = get_pareto_front(pfr, container=soln_set)
        _assign_crowding_dists_in_front(pareto_front, env, max_complexity)


def get_pareto_front(pareto_front_rank, container):
    res = [
        item for item in container
        if item.pareto_front_rank == pareto_front_rank
    ]
    assert len(res) > 0
    return res


def _assign_crowding_dists_in_front(pareto_front, env, max_complexity):
    """crowding-distance-assignment function from NSGA-II paper"""
    num_solns = len(pareto_front)
    for soln in pareto_front:
        soln.crowding_dist = 0
    for objective_func in (_perf_objective, _complexity_objective):
        # sort front in ascending order of objective val
        sorted_front = sorted(pareto_front,
                              key=lambda soln: objective_func(soln),
                              reverse=False)
        sorted_front[0].crowding_dist = math.inf
        sorted_front[(num_solns - 1)].crowding_dist = math.inf
        for idx in range(1, (num_solns - 1)):
            curr_soln = sorted_front[idx]
            neighbour_soln_left = sorted_front[idx - 1]
            neighbour_soln_right = sorted_front[idx + 1]
            fmax = _get_fmax(objective_func, env, max_complexity)
            fmin = _get_fmin(objective_func, env)
            curr_soln.crowding_dist += \
                (objective_func(neighbour_soln_right) -
                    objective_func(neighbour_soln_left)) / \
                (fmax - fmin)


def _get_fmax(objective_func, env, max_complexity):
    if objective_func == _perf_objective:
        return env.max_perf
    elif objective_func == _complexity_objective:
        return max_complexity
    else:
        raise Exception


def _get_fmin(objective_func, env):
    if objective_func == _perf_objective:
        return env.min_perf
    elif objective_func == _complexity_objective:
        return MIN_COMPLEXITY
    else:
        raise Exception


def select_parent_pop(gen_num, pop, parent_pop_size):
    """NSGA-II style selection of candidate parents: first sort by pareto front
    rank then sort by crowding dist for last front that can't fit fully in
    parent pop."""
    logging.info("Selecting parent pop")
    if gen_num == 0:
        assert len(pop) == parent_pop_size
        return pop
    else:
        crowded_comparison_sorted_pop = crowded_comparison_sort(pop)
        parent_pop = crowded_comparison_sorted_pop[0:parent_pop_size]
        return parent_pop


#        assert len(pop) >= parent_pop_size
#        pfrs = [indiv.pareto_front_rank for indiv in pop]
#        unique_pfrs = list(set(pfrs))
#        unique_pfrs_sorted_asc = sorted(unique_pfrs)
#
#        pfr_idx = 0
#        parent_pop = []
#        while not _parent_pop_is_full(parent_pop, parent_pop_size):
#            curr_pfr = unique_pfrs_sorted_asc[pfr_idx]
#            pareto_front = get_pareto_front(curr_pfr, container=pop)
#            parent_pop_slots_remaining = (parent_pop_size - len(parent_pop))
#            front_fully_fits = \
#                (len(pareto_front) <= parent_pop_slots_remaining)
#            if front_fully_fits:
#                parent_pop.extend(pareto_front)
#            else:
#                crowding_dist_sorted_desc = sorted(
#                    pareto_front,
#                    key=lambda indiv: indiv.crowding_dist,
#                    reverse=True)
#                indivs_to_add = \
#                    crowding_dist_sorted_desc[0:parent_pop_slots_remaining]
#                parent_pop.extend(indivs_to_add)
#            pfr_idx += 1
#        assert len(parent_pop) == parent_pop_size
#        return parent_pop


def crowded_comparison_sort(pop):
    # first sort by crowding dist desc, then by pfr asc.
    crowding_dist_desc = sorted(pop,
                                key=lambda indiv: indiv.crowding_dist,
                                reverse=True)
    return sorted(crowding_dist_desc,
                  key=lambda indiv: indiv.pareto_front_rank)


def crowded_comparison_operator(indiv_a, indiv_b):
    """NSGA-II crowded comparison operator: determines whether indiv_a is
    'better than' indiv_b in relation to Pareto front rank and crowding
    dist."""
    if indiv_a.pareto_front_rank < indiv_b.pareto_front_rank:
        return True
    elif indiv_a.pareto_front_rank == indiv_b.pareto_front_rank:
        return indiv_a.crowding_dist > indiv_b.crowding_dist
    elif indiv_a.pareto_front_rank > indiv_b.pareto_front_rank:
        return False
    else:
        raise Exception
