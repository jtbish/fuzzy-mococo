import argparse
import math

from lv_genotype import calc_num_lv_genes
from rb_genotype import calc_num_rb_genes

_MIN_NUM_MFS = 2


def parse_subspecies_tags(string):
    try:
        tags = string.split(" ")
        res = []
        for tag in tags:
            res.append(tuple(int(num_mfs) for num_mfs in tag.split(",")))
        return res
    except:
        raise argparse.ArgumentTypeError(string)


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


def get_subpop(pop, subspecies_tag):
    return [indiv for indiv in pop if indiv.subspecies_tag == subspecies_tag]
