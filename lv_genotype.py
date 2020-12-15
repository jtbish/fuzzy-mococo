from zadeh.domain import Domain
from zadeh.linguistic_var import LinguisticVar
from zadeh.membership_func import make_triangular_membership_func as make_tri
from subspecies import calc_num_lv_genes, calc_num_genes_for_mfs
from indiv import Indiv
import numpy as np

LV_ALLELE_MIN = 0.0
LV_ALLELE_MAX = 1.0


def make_lv_indiv(subspecies_tag):
    num_genes_needed = calc_num_lv_genes(subspecies_tag)
    genotype = np.random.uniform(low=LV_ALLELE_MIN,
                                 high=LV_ALLELE_MAX,
                                 size=num_genes_needed)
    return Indiv(subspecies_tag, genotype)


def make_ling_vars(subspecies_tag, genotype, env):
    num_genes_for_mfs = [calc_num_genes_for_mfs(num_mfs) for num_mfs in
                         subspecies_tag]
    _validate_genotype(genotype, num_genes_for_mfs)

    # build the ling vars by taking using appropriate genes for each
    ling_vars = []
    genotype_start_idx = 0
    for (obs_space_idx, num_genes_for_mf) in enumerate(num_genes_for_mfs):
        genotype_end_idx_excl = (genotype_start_idx + num_genes_for_mf)
        genes = genotype[genotype_start_idx:genotype_end_idx_excl]
        ling_vars.append(_make_ling_var(genes, num_mfs, env, obs_space_idx))
        genotype_start_idx = genotype_end_idx_excl
    made_it_to_genotype_end = (genotype_start_idx == len(genotype))
    assert made_it_to_genotype_end
    return tuple(ling_vars)


def _validate_genotype(genotype, num_genes_for_mfs):
    for allele in genotype:
        assert LV_ALLELE_MIN <= allele <= LV_ALLELE_MAX
    assert sum(num_genes_for_mfs) == len(genotype)


def _make_ling_var(genes, num_mfs, env, obs_space_idx):
    dim = env.obs_space[obs_space_idx]
    feature_name = dim.name
    domain = Domain(min=dim.lower, max=dim.upper)
    mfs = _make_tri_mfs(domain, genes, num_mfs, feature_name)
    return LinguisticVar(mfs, feature_name)


def _make_tri_mfs(domain, genes, num_membership_funcs, feature_name):
    relative_mults = genes
    # convert relative mults to absolute ref points in domain
    ref_points = [domain.min]
    for mult in relative_mults:
        last_ref_point = ref_points[-1]
        next_ref_point = \
            last_ref_point + mult*(domain.max - last_ref_point)
        assert domain.min <= next_ref_point <= domain.max
        ref_points.append(next_ref_point)
    ref_points.append(domain.max)
    assert len(ref_points) == num_membership_funcs

    membership_funcs = []
    # first mf
    membership_funcs.append(
        make_tri(domain, ref_points[0], ref_points[0], ref_points[1],
                 f"{feature_name}_0"))
    # middle mfs
    for i in range(0, (num_membership_funcs - 2)):
        membership_funcs.append(
            make_tri(domain, ref_points[i], ref_points[i + 1],
                     ref_points[i + 2], f"{feature_name}_{i+1}"))
    # last mf
    membership_funcs.append(
        make_tri(domain, ref_points[-2], ref_points[-1], ref_points[-1],
                 f"{feature_name}_{num_membership_funcs-1}"))
    assert len(membership_funcs) == num_membership_funcs
    return membership_funcs
