import copy
from collections import namedtuple

import numpy as np
from zadeh.aggregation import BoundedSumAggregation, WeightedAvgAggregation
from zadeh.error import UndefinedMappingError
from zadeh.inference_engine import InferenceEngine
from zadeh.logical_ops import (logical_and_min, logical_and_prod,
                               logical_or_max, logical_or_probor)
from zadeh.system import FuzzyRuleBasedSystem

InternalFitnessRecord = namedtuple("InternalFitnessRecord",
                                   ["genotype", "fitness"])
# indiv_idxs is 2-tuple giving idxs into (lv, rb) populations
# fitness = perf - alpha*frbs_complexity
JointFitnessRecord = namedtuple(
    "JointFitnessRecord",
    ["indiv_idxs", "frbs", "perf", "frbs_complexity", "fitness"])
SolnRecord = namedtuple(
    "SolnRecord",
    ["genotypes", "phenotypes", "frbs", "perf", "frbs_complexity", "fitness"])

ACTION_SET = {0, 2}
IE_CLASS_LABELS = (0, 2)
RB_ALLELE_SET = {-1, 0, 2}

MIN_PARETO_FRONT_RANK = 1
MIN_CROWDING_DIST = 0


def make_inference_engine(and_type, or_type, agg_type, env):
    class_labels = env.action_set

    if and_type == "min":
        logical_and_strat = logical_and_min
    elif and_type == "prod":
        logical_and_strat = logical_and_prod
    else:
        raise Exception

    if or_type == "max":
        logical_or_strat = logical_or_max
    elif or_type == "probor":
        logical_or_strat = logical_or_probor
    else:
        raise Exception

    if agg_type == "bsum":
        aggregation_strat = BoundedSumAggregation()
    elif agg_type == "wavg":
        aggregation_strat = WeightedAvgAggregation()
    else:
        raise Exception

    return InferenceEngine(class_labels=class_labels,
                           logical_and_strat=logical_and_strat,
                           logical_or_strat=logical_or_strat,
                           aggregation_strat=aggregation_strat)


def get_possibly_null_attr(obj, attr_name):
    attr = getattr(obj, attr_name)
    # accessing null attr should never be done so raise an error that will be
    # unhandled if it is
    if attr is None:
        raise ValueError
    else:
        return attr


def make_frbs(inference_engine, ling_vars, rule_base):
    return FuzzyRuleBasedSystem(inference_engine, ling_vars, rule_base)


def eval_perf(env, soln):
    # clone the env so that each eval has the same seq. of init obss
    env = copy.deepcopy(env)
    frbs = soln.frbs
    returns_agg_func = ENVS_CONFIG[env.name]["returns_agg_func"]
    perf_eval_func = ENVS_CONFIG[env.name]["perf_eval_func"]
    try:
        return perf_eval_func(env, frbs, returns_agg_func)
    except UndefinedMappingError:
        return ENVS_CONFIG[env.name]["perf_min"]


def _eval_mountain_car_perf(env, frbs, returns_agg_func):
    """Can possibly throw UndefinedMappingError if there is a hole in the frbs
    and cannot classify an input vec. In that case, let the caller deal with
    the error."""
    returns = []
    for seed in range(0, NUM_PERF_ROLLOUTS):
        input_vec = env.reset()
        return_ = 0
        for t in range(200):
            action = frbs.classify(input_vec)
            env_response = env.step(action)
            input_vec = env_response.obs
            return_ += env_response.reward
            if env_response.is_terminal:
                break
        returns.append(return_)
    perf = returns_agg_func(returns)
    return perf


def calc_lv_joint_fitness_records(lv_phenotypes, lv_idx, rb_phenotypes,
                                  rb_collabr_idxs, inference_engine,
                                  num_perf_rollouts, alpha):
    """Calcs and returns joint fitness records for a given individual of ling
    var population."""
    ling_vars = lv_phenotypes[lv_idx]
    rb_collabrs = [rb_phenotypes[idx] for idx in rb_collabr_idxs]
    joint_fitness_records = []
    for (rb_collabr_idx, rb_collabr) in zip(rb_collabr_idxs, rb_collabrs):
        rule_base = rb_collabr
        joint_fitness_records.append(
            _make_joint_fitness_record(inference_engine, ling_vars, rule_base,
                                       num_perf_rollouts, alpha, lv_idx,
                                       rb_collabr_idx))
    return joint_fitness_records


def calc_rb_joint_fitness_records(rb_phenotypes, rb_idx, lv_phenotypes,
                                  lv_collabr_idxs, inference_engine,
                                  num_perf_rollouts, alpha):
    """Calcs and returns joint fitness records for a given individual of
    rule base population."""
    rule_base = rb_phenotypes[rb_idx]
    lv_collabrs = [lv_phenotypes[idx] for idx in lv_collabr_idxs]
    joint_fitness_records = []
    for (lv_collabr_idx, lv_collabr) in zip(lv_collabr_idxs, lv_collabrs):
        ling_vars = lv_collabr
        joint_fitness_records.append(
            _make_joint_fitness_record(inference_engine, ling_vars, rule_base,
                                       num_perf_rollouts, alpha,
                                       lv_collabr_idx, rb_idx))
    return joint_fitness_records


def _make_joint_fitness_record(inference_engine, ling_vars, rule_base,
                               num_perf_rollouts, alpha, lv_idx, rb_idx):
    frbs = FuzzyRuleBasedSystem(inference_engine, ling_vars, rule_base)
    (perf, frbs_complexity,
     fitness) = _calc_joint_fitness(frbs, num_perf_rollouts, rule_base, alpha)
    record = JointFitnessRecord(indiv_idxs=(lv_idx, rb_idx),
                                frbs=frbs,
                                perf=perf,
                                frbs_complexity=frbs_complexity,
                                fitness=fitness)
    return record


def _calc_joint_fitness(frbs, num_perf_rollouts, rule_base, alpha):
    try:
        perf = _assess_perf(frbs, num_perf_rollouts)
    except UndefinedMappingError:
        perf = MIN_PERF
    frbs_complexity = frbs.calc_complexity()
    joint_fitness = perf - alpha * frbs_complexity
    return (perf, frbs_complexity, joint_fitness)


def assess_perf(frbs, num_perf_rollouts):
    """Can possibly throw UndefinedMappingError if there is a hole in the frbs
    and cannot classify an input vec. In that case, let the caller deal with
    the error."""
    returns = []
    for seed in range(0, num_perf_rollouts):
        env = mc_env_1(seed=seed,
                       normalise=NORMALISE_OBSS,
                       use_default_action_set=USE_DEFAULT_ACTION_SET)
        input_vec = env.reset()
        return_ = 0
        for t in range(200):
            action = frbs.classify(input_vec)
            env_response = env.step(action)
            input_vec = env_response.obs
            return_ += env_response.reward
            if env_response.is_terminal:
                break
        returns.append(return_)
    perf = np.mean(returns)
    return perf


def flatten_joint_fitness_mats(lv_joint_fitness_mat, rb_joint_fitness_mat):
    flat_joint_fitness_records = []
    for mat in (lv_joint_fitness_mat, rb_joint_fitness_mat):
        for jfrs in mat:
            flat_joint_fitness_records.extend(jfrs)
    return flat_joint_fitness_records


def make_best_soln_record(flat_joint_fitness_records, lv_pop, lv_phenotypes,
                          rb_pop, rb_phenotypes):
    best_jfr = max(flat_joint_fitness_records, key=lambda jfr: jfr.fitness)
    best_idxs = best_jfr.indiv_idxs
    lv_idx = best_idxs[0]
    rb_idx = best_idxs[1]
    best_genotypes = (lv_pop[lv_idx], rb_pop[rb_idx])
    best_phenotypes = (lv_phenotypes[lv_idx], rb_phenotypes[rb_idx])
    best_soln_record = SolnRecord(best_genotypes, best_phenotypes,
                                  best_jfr.frbs, best_jfr.perf,
                                  best_jfr.frbs_complexity, best_jfr.fitness)
    return best_soln_record


def make_internal_fitness_records(genotypes, joint_fitness_mat):
    # joint fitness mat contains list of joint fitness records, one set of
    # records for each genotype
    assert len(genotypes) == len(joint_fitness_mat)
    ifrs = []
    for (genotype, jfrs) in zip(genotypes, joint_fitness_mat):
        # internal fitness calced as maximum over joint tests
        internal_fitness = max([jfr.fitness for jfr in jfrs])
        ifrs.append(InternalFitnessRecord(genotype, internal_fitness))
    return ifrs
