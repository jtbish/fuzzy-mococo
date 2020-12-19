import copy
import numpy as np
import itertools
from collections import namedtuple

from zadeh.antecedent import CNFAntecedent
from zadeh.constants import CONSEQUENT_MAX, CONSEQUENT_MIN
from zadeh.rule import FuzzyRule
from zadeh.rule_base import FuzzyRuleBase
from indiv import Indiv

# housekeeping nt used for doing merges, not the actual CNF rule
# antecedent is binary activation mask for feature mfs, consequent is the
# action chosen
RuleRecord = namedtuple("RuleRecord", ["antecedent", "consequent"])

UNSPECIFIED_ALLELE = -1


def calc_num_rb_genes(subspecies_tag):
    return np.prod(subspecies_tag)


def make_rb_indiv(subspecies_tag, inference_engine):
    num_genes_needed = calc_num_rb_genes(subspecies_tag)
    possible_alleles = get_possible_rb_alleles(inference_engine)
    genotype = np.random.choice(possible_alleles, size=num_genes_needed)
    return Indiv(subspecies_tag, genotype)


def get_possible_rb_alleles(inference_engine):
    return [UNSPECIFIED_ALLELE] + list(inference_engine.class_labels)


def make_rule_base(subspecies_tag, genotype, inference_engine):
    possible_alleles = get_possible_rb_alleles(inference_engine)
    rule_records = _make_rule_records(subspecies_tag, genotype,
                                      possible_alleles)
    rules = _make_rules(rule_records, inference_engine)
    return FuzzyRuleBase(rules)


def _make_rule_records(subspecies_tag, genotype, possible_alleles):
    mf_idxs_all_dims = []
    for num_mfs in subspecies_tag:
        mf_idxs_curr_dim = list(range(0, num_mfs))
        mf_idxs_all_dims.append(mf_idxs_curr_dim)
    # itertools.product equiv to nested for loops, so *last* feature wraps
    # around the fastest in the linear genotype, second last wraps around
    # second fastest, etc.
    mf_idx_combos = list(itertools.product(*mf_idxs_all_dims))
    assert len(genotype) == len(mf_idx_combos)

    rule_records = []
    genotype_idx = 0
    for mf_idx_combo in mf_idx_combos:
        allele = genotype[genotype_idx]
        assert allele in possible_alleles
        if allele != UNSPECIFIED_ALLELE:
            antecedent = _make_rule_record_antecedent(mf_idx_combo,
                                                      subspecies_tag)
            rule_records.append(RuleRecord(antecedent, consequent=allele))
        genotype_idx += 1
    return rule_records


def _make_rule_record_antecedent(mf_idx_combo, subspecies_tag):
    antecedent = []
    for (spec_mf_idx, num_mfs) in zip(mf_idx_combo, subspecies_tag):
        binary_mf_activation_mask = [
            1 if spec_mf_idx == mf_idx else 0
            for mf_idx in range(0, num_mfs)
        ]
        assert binary_mf_activation_mask.count(1) == 1
        antecedent.append(binary_mf_activation_mask)
    return antecedent


def _make_rules(rule_records, inference_engine):
    rules = []
    for action in inference_engine.class_labels:
        records_with_action = [
            rr for rr in rule_records if rr.consequent == action
        ]
        merged_records = _merge_rule_records(records_with_action)
        rules.extend(_make_cnf_fuzzy_rules(merged_records, inference_engine))
    return rules


def _merge_rule_records(rule_records):
    prev_rule_records = rule_records
    done = False
    while not done:
        next_rule_records = copy.deepcopy(prev_rule_records)
        # iterate over records in order, using ith record as reference,
        # attempting to merge with records that come after it. don't use last
        # record as ref since nothing after it
        for idx in range(0, (len(prev_rule_records)-1)):
            ref_record = prev_rule_records[idx]
            records_to_compare = prev_rule_records[(idx + 1):]
            (merge_done, merge_idx, merged_record) = \
                _try_merge_ref_record_with_others(ref_record,
                                                  records_to_compare)
            if merge_done:
                # replace the ref record by the merged record and delete the
                # record that was merged
                next_rule_records[idx] = merged_record
                record_that_was_merged = records_to_compare[merge_idx]
                next_rule_records.remove(record_that_was_merged)
                break  # one merge per while loop iter allowed
        done = (next_rule_records == prev_rule_records)
        prev_rule_records = next_rule_records
    return prev_rule_records


def _try_merge_ref_record_with_others(ref_record, records_to_compare):
    assert len(records_to_compare) > 0
    merge_done = False
    merge_idx = None  # idx of record in records_to_compare that was merged
    merged_record = None
    for (idx, comp_record) in enumerate(records_to_compare):
        (merge_done,
         merged_record) = _try_merge_records(ref_record, comp_record)
        if merge_done:
            merge_idx = idx
            break  # stop on first successful merge
    return (merge_done, merge_idx, merged_record)


def _try_merge_records(ref_record, comp_record):
    ref_antecedent = ref_record.antecedent
    comp_antecedent = comp_record.antecedent

    # determine if the two antecedents can be merged
    can_merge = False
    assert len(ref_antecedent) == len(comp_antecedent)
    for (ref_part, comp_part) in zip(ref_antecedent, comp_antecedent):
        assert len(ref_part) == len(comp_part)
        if ref_part == comp_part:
            # common part of both antecedents that can be factored out
            # so merge is possible
            can_merge = True
            break

    merged_record = None
    if can_merge:
        assert ref_record.consequent == comp_record.consequent
        consequent = ref_record.consequent
        # do the merge by bitwise ORing each part of the antecedents
        new_antecedent = []
        for (ref_part, comp_part) in zip(ref_antecedent, comp_antecedent):
            new_part = [int(r or c) for (r, c) in zip(ref_part, comp_part)]
            new_antecedent.append(new_part)
        merged_record = RuleRecord(new_antecedent, consequent)

    merge_done = can_merge
    return (merge_done, merged_record)


def _make_cnf_fuzzy_rules(merged_rule_records, inference_engine):
    rules = []
    for rr in merged_rule_records:
        antecedent = CNFAntecedent(membership_func_usages=rr.antecedent)
        consequent = \
            _make_consequent(class_labels=inference_engine.class_labels,
                             selected_label=rr.consequent)
        rule = FuzzyRule(antecedent, consequent)
        rules.append(rule)
    return rules


def _make_consequent(class_labels, selected_label):
    consequent = {}
    for class_label in class_labels:
        if class_label == selected_label:
            consequent[class_label] = CONSEQUENT_MAX
        else:
            consequent[class_label] = CONSEQUENT_MIN
    return consequent
