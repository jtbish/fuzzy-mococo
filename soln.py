from util import get_possibly_null_attr
from mo_constants import (MIN_PARETO_FRONT_RANK, MIN_CROWDING_DIST,
                          MIN_DOMINATION_COUNT)
from object_creation import get_next_soln_id


class Solution:
    """Data type to represent joint solution in MO space."""
    def __init__(self, lv_indiv, rb_indiv, frbs):
        self._lv_indiv = lv_indiv
        self._rb_indiv = rb_indiv
        self._frbs = frbs
        assert lv_indiv.subspecies_tag == rb_indiv.subspecies_tag
        self._subspecies_tag = lv_indiv.subspecies_tag
        self._perf = None
        self._complexity = None
        self._pareto_front_rank = None
        self._crowding_dist = None
        self._domination_count = None
        self._dominated_set = None
        self._soln_id = get_next_soln_id()

    @property
    def lv_indiv(self):
        return self._lv_indiv

    @property
    def rb_indiv(self):
        return self._rb_indiv

    @property
    def frbs(self):
        return self._frbs

    @property
    def subspecies_tag(self):
        return self._subspecies_tag

    @property
    def perf(self):
        return get_possibly_null_attr(self, "_perf")

    @perf.setter
    def perf(self, val):
        self._perf = val

    @property
    def complexity(self):
        return get_possibly_null_attr(self, "_complexity")

    @complexity.setter
    def complexity(self, val):
        self._complexity = val

    @property
    def pareto_front_rank(self):
        return get_possibly_null_attr(self, "_pareto_front_rank")

    @pareto_front_rank.setter
    def pareto_front_rank(self, val):
        assert val >= MIN_PARETO_FRONT_RANK
        self._pareto_front_rank = val

    @property
    def crowding_dist(self):
        return get_possibly_null_attr(self, "_crowding_dist")

    @crowding_dist.setter
    def crowding_dist(self, val):
        assert val >= MIN_CROWDING_DIST
        self._crowding_dist = val

    @property
    def domination_count(self):
        """Number of solns *this soln is dominated by*"""
        return get_possibly_null_attr(self, "_domination_count")

    @domination_count.setter
    def domination_count(self, val):
        assert val >= MIN_DOMINATION_COUNT
        self._domination_count = val

    @property
    def dominated_set(self):
        """The set of solns that *this soln dominates*"""
        return get_possibly_null_attr(self, "_dominated_set")

    @dominated_set.setter
    def dominated_set(self, val):
        self._dominated_set = val

    def does_contain(self, indiv):
        return indiv == self._lv_indiv or indiv == self._rb_indiv

    def __eq__(self, other):
        # use global soln id for equality because may have duplicate indiv
        # genotypes contained in two solns but want to treat them as separate
        # solns
        return self._soln_id == other._soln_id
