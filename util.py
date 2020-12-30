from zadeh.aggregation import BoundedSumAggregation, WeightedAvgAggregation
from zadeh.inference_engine import InferenceEngine
from zadeh.logical_ops import (logical_and_min, logical_and_prod,
                               logical_or_max, logical_or_probor)
from zadeh.system import FuzzyRuleBasedSystem
from collections import namedtuple


PopRecord = namedtuple("PopRecord", ["parents", "children"])


def make_inference_engine(and_type, or_type, agg_type, env):
    class_labels = env.action_set

    if and_type == "min":
        logical_and_strat = logical_and_min
    elif and_type == "prod":
        logical_and_strat = logical_and_prod
    else:
        assert False

    if or_type == "max":
        logical_or_strat = logical_or_max
    elif or_type == "probor":
        logical_or_strat = logical_or_probor
    else:
        assert False

    if agg_type == "bsum":
        aggregation_strat = BoundedSumAggregation()
    elif agg_type == "wavg":
        aggregation_strat = WeightedAvgAggregation()
    else:
        assert False

    return InferenceEngine(class_labels, logical_and_strat, logical_or_strat,
                           aggregation_strat)


def make_frbs(inference_engine, ling_vars, rule_base):
    return FuzzyRuleBasedSystem(inference_engine, ling_vars, rule_base)


def get_possibly_null_attr(obj, attr_name):
    attr = getattr(obj, attr_name)
    # accessing null attr should never be done so raise an error that will be
    # unhandled if it is
    if attr is None:
        raise ValueError("attr is null")
    else:
        return attr
