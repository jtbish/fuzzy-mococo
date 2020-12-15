import argparse
import math

import numpy as np

_MIN_NUM_MFS = 3


def parse_subspecies_tag(string):
    try:
        return tuple(int(num_mfs) for num_mfs in string.split(","))
    except:
        raise argparse.ArgumentTypeError()


def validate_subspecies_tags(subspecies_tags, env):
    num_features = len(env.obs_space)
    for subspecies_tag in subspecies_tags:
        assert len(subspecies_tag) == num_features
        for num_mfs in subspecies_tag:
            assert num_mfs >= _MIN_NUM_MFS


def make_pop_init_pmfs(subspecies_tags):
    total_lv_genes = sum(
        [calc_num_lv_genes(subspecies_tag) for subspecies_tag in
         subspecies_tags])
    total_rb_genes = sum(
        [calc_num_rb_genes(subspecies_tag) for subspecies_tag in
         subspecies_tags])

    lv_init_pmf = {}
    for subspecies_tag in subspecies_tags:
        lv_init_pmf[subspecies_tag] = \
            (calc_num_lv_genes(subspecies_tag)) / total_lv_genes
    rb_init_pmf = {}
    for subspecies_tag in subspecies_tags:
        rb_init_pmf[subspecies_tag] = \
            (calc_num_rb_genes(subspecies_tag)) / total_rb_genes

    pmf_sum = 1.0
    assert math.isclose(sum(lv_init_pmf.values()), pmf_sum)
    assert math.isclose(sum(rb_init_pmf.values()), pmf_sum)
    return (lv_init_pmf, rb_init_pmf)


def calc_num_lv_genes(subspecies_tag):
    return sum([calc_num_genes_for_mfs(num_mfs) for num_mfs in
               subspecies_tag])


def calc_num_genes_for_mfs(num_mfs):
    return num_mfs - 2


def calc_num_rb_genes(subspecies_tag):
    return np.prod(subspecies_tag)


def get_subpop(pop, subspecies_tag):
    return [indiv for indiv in pop if indiv.subspecies_tag == subspecies_tag]
