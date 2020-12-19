import numpy as np
import copy
from lv_genotype import LV_ALLELE_MIN, LV_ALLELE_MAX
from rb_genotype import get_possible_rb_alleles
from multi_objective import crowded_comparison_operator
from subspecies import get_subpop


def run_lv_ga(lv_parent_pop, child_pop_size, tourn_size, p_cross_line,
              mut_sigma):
    return _run_ga(lv_parent_pop, child_pop_size, tourn_size,
                   cross_callback={
                       "func": _line_recombination,
                       "kwargs": {
                           "p_cross_line": p_cross_line
                       }
                   },
                   mut_callback={
                       "func": _gaussian_mutation,
                       "kwargs": {
                           "sigma": mut_sigma
                       }
                   })


def run_rb_ga(rb_parent_pop, child_pop_size, tourn_size, p_cross_swap,
              p_mut_flip, inference_engine):
    possible_rb_alleles = get_possible_rb_alleles(inference_engine)
    return _run_ga(rb_parent_pop, child_pop_size, tourn_size,
                   cross_callback={
                       "func": _uniform_crossover,
                       "kwargs": {
                           "p_cross_swap": p_cross_swap
                       }
                   },
                   mut_callback={
                       "func": _flip_mutation,
                       "kwargs": {
                           "p_mut_flip": p_mut_flip,
                           "possible_rb_alleles":  possible_rb_alleles
                       }
                   })


def _run_ga(parent_pop, child_pop_size, tourn_size, cross_callback,
            mut_callback):
    """Vanilla GA with parametrised crossover and mutation funcs."""
    assert child_pop_size % 2 == 0
    child_pop = []
    for _ in range(int(child_pop_size / 2)):
        parent_a = _first_selection(parent_pop, tourn_size)
        parent_b = _second_selection(parent_pop, tourn_size, parent_a)
        children = _crossover_and_mutate(parent_a, parent_b, cross_callback,
                                         mut_callback)
        for child in children:
            child_pop.append(child)
    assert len(child_pop) == child_pop_size
    return child_pop


def _first_selection(parent_pop, tourn_size):
    return _tournament_selection(parent_pop, tourn_size)


def _second_selection(parent_pop, tourn_size, parent_a):
    # select parent_b from same subpop as parent_a to ensure that they can
    # crossover ok
    parent_subpop = get_subpop(pop=parent_pop,
                               subspecies_tag=parent_a.subspecies_tag)
    return _tournament_selection(parent_subpop, tourn_size)


def _tournament_selection(pop, tourn_size):
    def _select_random_indiv(pop):
        return np.random.choice(pop)
    best_indiv = _select_random_indiv(pop)
    for _ in range(2, (tourn_size + 1)):
        next_indiv = _select_random_indiv(pop)
        if crowded_comparison_operator(next_indiv, best_indiv):
            best_indiv = next_indiv
    return best_indiv


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


def _flip_mutation(child, p_mut_flip, possible_rb_alleles):
    flip_map = _build_flip_map(possible_rb_alleles)
    for idx in range(0, len(child)):
        should_flip = np.random.rand() < p_mut_flip
        if should_flip:
            curr_val = child[idx]
            flip_options = flip_map[curr_val]
            # all options equally weighted
            new_val = np.random.choice(flip_options)
            child[idx] = new_val


def _build_flip_map(possible_rb_alleles):
    flip_map = {}
    for curr_allele in possible_rb_alleles:
        other_alleles = copy.deepcopy(possible_rb_alleles)
        other_alleles.remove(curr_allele)
        flip_map[curr_allele] = other_alleles
    return flip_map


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
