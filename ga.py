import copy
import logging

import numpy as np

from indiv import Indiv
from lv_genotype import LV_ALLELE_MAX, LV_ALLELE_MIN
from multi_objective import crowded_comparison_operator
from rb_genotype import get_possible_rb_alleles, repair_rb_genotype_if_needed
from subspecies import get_subpop, sample_subspecies_tag

_GAUSSIAN_MUTATION_MU = 0.0


def run_lv_ga(lv_parent_pop, child_pop_size, subspecies_pmf, tourn_size,
              p_cross_line, mut_sigma):
    logging.info("Running lv ga")
    return _run_ga(lv_parent_pop,
                   child_pop_size,
                   subspecies_pmf,
                   tourn_size,
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


def run_rb_ga(rb_parent_pop, child_pop_size, subspecies_pmf, tourn_size,
              cross_swap_mult, mut_flip_mult,
              inference_engine):
    logging.info("Running rb ga")
    return _run_ga(rb_parent_pop,
                   child_pop_size,
                   subspecies_pmf,
                   tourn_size,
                   cross_callback={
                       "func": _uniform_crossover,
                       "kwargs": {
                           "cross_swap_mult": cross_swap_mult
                       }
                   },
                   mut_callback={
                       "func": _flip_mutation,
                       "kwargs": {
                           "mut_flip_mult": mut_flip_mult,
                           "inference_engine": inference_engine
                       }
                   })


def _run_ga(parent_pop, child_pop_size, subspecies_pmf, tourn_size,
            cross_callback, mut_callback):
    """Vanilla GA with parametrised crossover and mutation funcs."""
    assert child_pop_size % 2 == 0
    child_pop = []
    for _ in range(child_pop_size // 2):
        subspecies_tag = sample_subspecies_tag(subspecies_pmf)
        subpop = get_subpop(parent_pop, subspecies_tag)
        parent_a = _tournament_selection(subpop, tourn_size)
        parent_b = _tournament_selection(subpop, tourn_size)
        children = _crossover_and_mutate(parent_a, parent_b, cross_callback,
                                         mut_callback)
        for child in children:
            child_pop.append(child)
    assert len(child_pop) == child_pop_size
    return child_pop


def _tournament_selection(subpop, tourn_size):
    def _select_random_indiv(subpop):
        return np.random.choice(subpop)
    best_indiv = _select_random_indiv(subpop)
    for _ in range(2, (tourn_size + 1)):
        next_indiv = _select_random_indiv(subpop)
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


def _uniform_crossover(child_a_genotype, child_b_genotype, cross_swap_mult):
    assert len(child_a_genotype) == len(child_b_genotype)
    num_genes = len(child_a_genotype)
    assert 1 <= cross_swap_mult <= num_genes
    p_cross_swap = cross_swap_mult * (1 / num_genes)
    for idx in range(0, num_genes):
        should_swap = np.random.rand() < p_cross_swap
        if should_swap:
            child_a_genotype[idx], child_b_genotype[idx] = \
                child_b_genotype[idx], child_a_genotype[idx]


def _line_recombination(child_a_genotype, child_b_genotype, p_cross_line):
    """Essentials of Metaheuristics Algorithm 28."""
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


def _flip_mutation(child_genotype, mut_flip_mult, inference_engine):
    possible_rb_alleles = get_possible_rb_alleles(inference_engine)
    flip_map = _build_flip_map(possible_rb_alleles)
    num_genes = len(child_genotype)
    assert 1 <= mut_flip_mult <= num_genes
    p_mut_flip = mut_flip_mult * (1 / num_genes)
    for idx in range(0, num_genes):
        should_flip = np.random.rand() < p_mut_flip
        if should_flip:
            curr_val = child_genotype[idx]
            flip_options = flip_map[curr_val]
            # all options equally weighted
            new_val = np.random.choice(flip_options)
            child_genotype[idx] = new_val
    repair_rb_genotype_if_needed(child_genotype, inference_engine)


def _build_flip_map(possible_rb_alleles):
    flip_map = {}
    for curr_allele in possible_rb_alleles:
        other_alleles = [
            allele for allele in possible_rb_alleles if allele != curr_allele
        ]
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
