import argparse
import numpy as np
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


def make_subspecies_pmfs_both_pops(subspecies_tags, subspecies_pmf_base):
    # subspecies pmfs proportional to base^(num dims)
    assert subspecies_pmf_base >= 1.0
    base = subspecies_pmf_base

    lv_denom = sum([base**calc_num_lv_genes(subspecies_tag) for
                    subspecies_tag in subspecies_tags])
    lv_subspecies_pmf = {}
    for subspecies_tag in subspecies_tags:
        numer = base**calc_num_lv_genes(subspecies_tag)
        lv_subspecies_pmf[subspecies_tag] = (numer / lv_denom)

    rb_denom = sum([base**calc_num_rb_genes(subspecies_tag) for
                   subspecies_tag in subspecies_tags])
    rb_subspecies_pmf = {}
    for subspecies_tag in subspecies_tags:
        numer = base**calc_num_rb_genes(subspecies_tag)
        rb_subspecies_pmf[subspecies_tag] = (numer / rb_denom)

    pmf_sum = 1.0
    assert math.isclose(sum(lv_subspecies_pmf.values()), pmf_sum)
    assert math.isclose(sum(rb_subspecies_pmf.values()), pmf_sum)
    return (lv_subspecies_pmf, rb_subspecies_pmf)


def get_subpop(pop, subspecies_tag):
    return [indiv for indiv in pop if indiv.subspecies_tag == subspecies_tag]


def sample_subspecies_tags(subspecies_pmf, sample_size):
    choices = np.empty(shape=len(subspecies_pmf), dtype=object)
    for (idx, choice) in enumerate(subspecies_pmf.keys()):
        choices[idx] = choice
    probs = list(subspecies_pmf.values())
    return list(np.random.choice(a=choices,
                                 size=sample_size,
                                 p=probs))


def sample_subspecies_tag(subspecies_pmf):
    res_ls = sample_subspecies_tags(subspecies_pmf, sample_size=1)
    assert len(res_ls) == 1
    return res_ls[0]
