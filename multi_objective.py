import copy
import math


FMIN_PERF = -200.0
FMAX_PERF = ?
FMIN_COMPLEXITY = 0
FMAX_COMPLEXITY = ?


def assign_pareto_front_ranks(soln_set):
    """Second part of fast-non-dominated-sort algorithm from NSGA-II paper,
    except not explicitly storing and returning fronts but rather just storing
    front ranks of each soln."""
    _find_dominations(soln_set)

    # find F1
    first_front = []
    for soln in soln_set:
        if soln.domination_count == 0:
            soln.pareto_front_rank = 1
            first_front.append(soln)

    # find subsequent fronts
    curr_front_number = 1
    curr_front = first_front
    while len(curr_front) != 0:
        next_front = []
        for soln in curr_front:
            for dominated_soln in soln.dominated_set:
                dominated_soln.domination_count -= 1
                if dominated_soln.domination_count == 0:
                    dominated_soln.pareto_front_rank = curr_front_number + 1
                    next_front.append(dominated_soln)
        curr_front_number += 1
        curr_front = next_front


def _find_dominations(soln_set):
    """First part of fast-non-dominated-sort algorithm from NSGA-II paper"""
    for soln in soln_set:
        soln.dominated_set = []
        soln.domination_count = 0

        other_solns = copy.deepcopy(soln_set)
        other_solns.remove(soln)
        for other_soln in other_solns:
            if _does_dominate(soln, other_soln):
                soln.dominated_set = soln.dominated_set + [other_soln]
            elif _does_dominate(other_soln, soln):
                soln.domination_count += 1


def _does_dominate(soln_a, soln_b):
    """Does soln_a dominate soln_b?
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


def assign_crowding_dists(soln_set):
    soln_pfrs = [soln.pareto_front_rank for soln in soln_set]
    min_pfr = min(soln_pfrs)
    assert min_pfr == 1
    max_pfr = max(soln_pfrs)
    for pareto_front_rank in range(min_pfr, (max_pfr + 1)):
        pareto_front = _get_pareto_front(soln_set, pareto_front_rank)
        _assign_crowding_dists_in_front(pareto_front)


def _get_pareto_front(soln_set, pareto_front_rank):
    return [
        soln for soln in soln_set
        if soln.pareto_front_rank == pareto_front_rank
    ]


def _assign_crowding_dists_in_front(pareto_front):
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
            fmax = _get_fmax(objective_func)
            fmin = _get_fmin(objective_func)
            curr_soln.crowding_dist += \
                (objective_func(neighbour_soln_right) -
                    objective_func(neighbour_soln_left)) / \
                (fmax - fmin)


def _get_fmax(objective_func):
    if objective_func == _perf_objective:
        return FMAX_PERF
    elif objective_func == _complexity_objective:
        return FMAX_COMPLEXITY
    else:
        raise Exception


def _get_fmin(objective_func):
    if objective_func == _perf_objective:
        return FMIN_PERF
    elif objective_func == _complexity_objective:
        return FMIN_COMPLEXITY
    else:
        raise Exception
