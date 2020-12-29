import logging
import math

import numpy as np
from zadeh.error import UndefinedMappingError
from rb_genotype import calc_rb_genotype_complexity

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
    rb_genotype = soln.rb_indiv.genotype
    return calc_rb_genotype_complexity(rb_genotype)


def calc_max_complexity(subspecies_tags):
    max_possible_fuzzy_decision_regions = \
        max([np.prod(subspecies_tag) for subspecies_tag in subspecies_tags])
    return max_possible_fuzzy_decision_regions


def assign_pareto_front_ranks(container):
    """Second part of fast-non-dominated-sort algorithm from NSGA-II paper,
    except not explicitly storing and returning fronts but rather just storing
    front ranks of each item in container."""
    logging.info("Assigning pareto front ranks")
    _find_dominations(container)

    # find first front
    first_front = []
    for item in container:
        if item.domination_count == 0:
            item.pareto_front_rank = 1
            first_front.append(item)

    # find subsequent fronts
    curr_front_number = 1
    curr_front = first_front
    while len(curr_front) != 0:
        logging.debug(f"Trying to find front number {curr_front_number + 1}")
        next_front = []
        for item in curr_front:
            for dominated_item in item.dominated_set:
                dominated_item.domination_count -= 1
                if dominated_item.domination_count == 0:
                    dominated_item.pareto_front_rank = curr_front_number + 1
                    next_front.append(dominated_item)
        curr_front_number += 1
        curr_front = next_front
    assert np.all([item.domination_count == 0 for item in container])


def _find_dominations(container):
    """First part of fast-non-dominated-sort algorithm from NSGA-II paper"""
    _init_container_domination_attrs(container)
    for item in container:
        other_items = [item_ for item_ in container if item_ != item]
        for other_item in other_items:
            if _does_dominate(item, other_item):
                item.dominated_set = item.dominated_set + [other_item]
            elif _does_dominate(other_item, item):
                item.domination_count += 1


def _init_container_domination_attrs(container):
    for item in container:
        item.dominated_set = []
        item.domination_count = MIN_DOMINATION_COUNT


def _does_dominate(item_a, item_b):
    """Does item_a Pareto dominate item_b?
    To dominate item_b, item_a must be strictly better in at least one
    objective and no worse in all others.

    Essentials of Metaheuristics Alg. 98: Pareto Domination"""
    res = False
    for objective_func in (_perf_objective, _complexity_objective):
        if _is_better(objective_func, item_a, item_b):
            res = True
        elif _is_better(objective_func, item_b, item_a):
            return False
    return res


def _perf_objective(item):
    return item.perf


def _complexity_objective(item):
    return item.complexity


def _is_better(objective_func, first_item, second_item):
    """Is first_item 'better than' second_item according to objective_func?"""
    if objective_func == _perf_objective:
        # higher is better
        return objective_func(first_item) > objective_func(second_item)
    elif objective_func == _complexity_objective:
        # lower is better
        return objective_func(first_item) < objective_func(second_item)
    else:
        raise Exception


def assign_crowding_dists(container, env, max_complexity):
    logging.info("Assigning crowding dists")
    pfrs = [item.pareto_front_rank for item in container]
    min_pfr = min(pfrs)
    assert min_pfr == MIN_PARETO_FRONT_RANK
    max_pfr = max(pfrs)
    for pfr in range(min_pfr, (max_pfr + 1)):
        pareto_front = get_pareto_front(pfr, container)
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
    num_items = len(pareto_front)
    for item in pareto_front:
        item.crowding_dist = 0
    for objective_func in (_perf_objective, _complexity_objective):
        # sort front in ascending order of objective val
        sorted_front = sorted(pareto_front,
                              key=lambda item: objective_func(item),
                              reverse=False)
        sorted_front[0].crowding_dist = math.inf
        sorted_front[(num_items - 1)].crowding_dist = math.inf
        for idx in range(1, (num_items - 1)):
            curr_item = sorted_front[idx]
            neighbour_item_left = sorted_front[idx - 1]
            neighbour_item_right = sorted_front[idx + 1]
            fmax = _get_fmax(objective_func, env, max_complexity)
            fmin = _get_fmin(objective_func, env)
            curr_item.crowding_dist += \
                (objective_func(neighbour_item_right) -
                    objective_func(neighbour_item_left)) / \
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


def select_parent_pop(pop, parent_pop_size, env, max_complexity):
    """NSGA-II style selection of candidate parents: first sort by pareto front
    rank then sort by crowding dist for last front that can't fit fully in
    parent pop. Selecting parents equiv. to updating archive."""
    logging.info("Selecting parent pop / updating archive")
    assign_pareto_front_ranks(pop)
    assign_crowding_dists(pop, env, max_complexity)
    crowded_comparison_sorted_pop = crowded_comparison_sort(pop)
    parent_pop = crowded_comparison_sorted_pop[0:parent_pop_size]
    return parent_pop


def crowded_comparison_sort(container):
    # first sort by crowding dist desc, then by pfr asc.
    crowding_dist_desc = sorted(container,
                                key=lambda item: item.crowding_dist,
                                reverse=True)
    return sorted(crowding_dist_desc,
                  key=lambda item: item.pareto_front_rank)


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
