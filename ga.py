import numpy as np
import copy
from lv_genotype import LV_ALLELE_MIN, LV_ALLELE_MAX
from rb_genotype import get_possible_rb_alleles
from multi_objective import crowded_comparison_operator
from subspecies import get_subpop
from indiv import Indiv

_GAUSSIAN_MUTATION_MU = 0.0


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
    child_a_genotype = copy.deepcopy(parent_a.genotype)
    child_b_genotype = copy.deepcopy(parent_b.genotype)
    _crossover_children(child_a_genotype, child_b_genotype, cross_callback)
    _mutate_child(child_a_genotype, mut_callback)
    _mutate_child(child_b_genotype, mut_callback)

    assert parent_a.subspecies_tag == parent_b.subspecies_tag
    subspecies_tag = parent_a.subspecies_tag
    child_a = Indiv(subspecies_tag, child_a_genotype)
    child_b = Indiv(subspecies_tag, child_b_genotype)
    return (child_a, child_b)


def _crossover_children(child_a_genotype, child_b_genotype, cross_callback):
    cross_func = cross_callback["func"]
    func_kwargs = cross_callback["kwargs"]
    cross_func(child_a_genotype, child_b_genotype, **func_kwargs)


def _uniform_crossover(child_a_genotype, child_b_genotype, p_cross_swap):
    assert len(child_a_genotype) == len(child_b_genotype)
    for idx in range(0, len(child_a_genotype)):
        should_swap = np.random.rand() < p_cross_swap
        if should_swap:
            child_a_genotype[idx], child_b_genotype[idx] = \
                child_b_genotype[idx], child_a_genotype[idx]


def _line_recombination(child_a_genotype, child_b_genotype, p_cross_line):
    assert len(child_a_genotype) == len(child_b_genotype)
    should_cross = np.random.rand() < p_cross_line
    if should_cross:
        alpha = np.random.rand()
        beta = np.random.rand()
        for idx in range(0, len(child_a_genotype)):
            a_i = child_a_genotype[idx]
            b_i = child_b_genotype[idx]
            t = alpha * a_i + (1 - alpha) * b_i
            s = beta * b_i + (1 - beta) * a_i
            assert LV_ALLELE_MIN <= t <= LV_ALLELE_MAX
            assert LV_ALLELE_MIN <= s <= LV_ALLELE_MAX
            child_a_genotype[idx] = t
            child_b_genotype[idx] = s


def _mutate_child(child_genotype, mut_callback):
    mut_func = mut_callback["func"]
    func_kwargs = mut_callback["kwargs"]
    mut_func(child_genotype, **func_kwargs)


def _flip_mutation(child_genotype, p_mut_flip, possible_rb_alleles):
    flip_map = _build_flip_map(possible_rb_alleles)
    for idx in range(0, len(child_genotype)):
        should_flip = np.random.rand() < p_mut_flip
        if should_flip:
            curr_val = child_genotype[idx]
            flip_options = flip_map[curr_val]
            # all options equally weighted
            new_val = np.random.choice(flip_options)
            child_genotype[idx] = new_val


def _build_flip_map(possible_rb_alleles):
    flip_map = {}
    for curr_allele in possible_rb_alleles:
        other_alleles = copy.deepcopy(possible_rb_alleles)
        other_alleles.remove(curr_allele)
        flip_map[curr_allele] = other_alleles
    return flip_map


def _gaussian_mutation(child_genotype, sigma):
    for idx in range(0, len(child_genotype)):
        curr_val = child_genotype[idx]
        noise = np.random.normal(loc=_GAUSSIAN_MUTATION_MU, scale=sigma)
        new_val = curr_val + noise
        if new_val < LV_ALLELE_MIN or new_val > LV_ALLELE_MAX:
            # mirror the noise
            new_val = curr_val - noise
        assert LV_ALLELE_MIN <= new_val <= LV_ALLELE_MAX
        child_genotype[idx] = new_val
