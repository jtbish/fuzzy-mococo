import math

import numpy as np
from zadeh.domain import Domain
from zadeh.linguistic_var import LinguisticVar
from zadeh.membership_func import make_triangular_membership_func as make_tri
from zadeh.membership_func import make_trapezoidal_membership_func as make_trap

from indiv import Indiv

LV_ALLELE_MIN = 0.0
LV_ALLELE_MAX = 1.0

SUBDOMAIN_VALID_MULT = 0.75


def calc_num_lv_genes(subspecies_tag):
    return sum([calc_num_genes_for_mfs(num_mfs) for num_mfs in subspecies_tag])


def calc_num_genes_for_mfs(num_mfs):
    # encoding the end traps plus the middle tris in the genotype, so same
    # number of genes as num_mfs
    return num_mfs


def make_lv_indiv(subspecies_tag):
    num_genes_needed = calc_num_lv_genes(subspecies_tag)
    genotype = np.random.uniform(low=LV_ALLELE_MIN,
                                 high=LV_ALLELE_MAX,
                                 size=num_genes_needed)
    return Indiv(subspecies_tag, genotype)


def make_ling_vars(subspecies_tag, genotype, env):
    num_genes_for_mfss = [
        calc_num_genes_for_mfs(num_mfs) for num_mfs in subspecies_tag
    ]
    _validate_genotype(genotype, num_genes_for_mfss)

    # build the ling vars by using appropriate genes for each
    ling_vars = []
    genotype_start_idx = 0
    for (obs_space_idx, (num_genes_for_mfs, num_mfs)) in \
            enumerate(zip(num_genes_for_mfss, subspecies_tag)):
        genotype_end_idx_excl = (genotype_start_idx + num_genes_for_mfs)
        genes = genotype[genotype_start_idx:genotype_end_idx_excl]
        ling_vars.append(_make_ling_var(genes, num_mfs, env, obs_space_idx))
        genotype_start_idx = genotype_end_idx_excl
    made_it_to_genotype_end = (genotype_start_idx == len(genotype))
    assert made_it_to_genotype_end
    return tuple(ling_vars)


def _validate_genotype(genotype, num_genes_for_mfss):
    for allele in genotype:
        assert LV_ALLELE_MIN <= allele <= LV_ALLELE_MAX
    assert sum(num_genes_for_mfss) == len(genotype)


def _make_ling_var(genes, num_mfs, env, obs_space_idx):
    dim = env.obs_space[obs_space_idx]
    feature_name = dim.name
    domain = Domain(min=dim.lower, max=dim.upper)
    mfs = _make_mfs(domain, genes, num_mfs, feature_name)
    return LinguisticVar(mfs, feature_name)


def _make_mfs(domain, genes, num_mfs, feature_name):
    subdomains = _make_subdomains(domain, num_mfs)
    ref_points = _make_ref_points(domain, subdomains, genes, num_mfs)

    membership_funcs = []
    # first mf: trap
    membership_funcs.append(
        make_trap(domain, ref_points[0], ref_points[0], ref_points[1],
                  ref_points[2], f"{feature_name}_0"))
    # middle mfs: tris
    for i in range(1, ((num_mfs - 2)+1)):
        membership_funcs.append(
            make_tri(domain, ref_points[i], ref_points[i + 1],
                     ref_points[i + 2], f"{feature_name}_{i+1}"))
    # last mf: trap
    membership_funcs.append(
        make_trap(domain, ref_points[-3], ref_points[-2], ref_points[-1],
                  ref_points[-1], f"{feature_name}_{num_mfs-1}"))
    assert len(membership_funcs) == num_mfs
    return membership_funcs


def _make_subdomains(domain, num_mfs):
    num_subdomains = num_mfs
    subdomain_size_max = ((domain.max - domain.min) / num_subdomains)

    subdomains = []
    lhs_ref_point = domain.min
    assert 0 < SUBDOMAIN_VALID_MULT <= 1.0
    for _ in range(num_subdomains):
        rhs_ref_point = (lhs_ref_point + subdomain_size_max)
        both_sides_offset_mult = ((1.0 - SUBDOMAIN_VALID_MULT) / 2)
        both_sides_offset_abs = (both_sides_offset_mult * subdomain_size_max)
        subdomain_min = (lhs_ref_point + both_sides_offset_abs)
        subdomain_max = (rhs_ref_point - both_sides_offset_abs)
        subdomains.append(Domain(min=subdomain_min, max=subdomain_max))
        lhs_ref_point = rhs_ref_point
    assert math.isclose(rhs_ref_point, domain.max)
    assert len(subdomains) == num_subdomains
    return subdomains


def _make_ref_points(domain, subdomains, genes, num_mfs):
    # construct the ref points in the domain used to build the mfs
    assert len(subdomains) == len(genes)
    ref_points = [domain.min]
    for (subdomain, gene) in zip(subdomains, genes):
        mult = gene
        subdomain_size = (subdomain.max - subdomain.min)
        ref_point = (subdomain.min + (mult * subdomain_size))
        ref_points.append(ref_point)
    ref_points.append(domain.max)
    assert len(ref_points) == (num_mfs + 2)
#    if num_mfs == 2:
#        print(f"{domain}: {subdomains}: {genes} -> {ref_points}")
    return ref_points
