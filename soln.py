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
        if self._perf is None:
            raise ValueError
        else:
            return self._perf

    @perf.setter
    def perf(self, val):
        self._perf = val

    @property
    def complexity(self):
        if self._complexity is None:
            raise ValueError
        else:
            return self._complexity

    @complexity.setter
    def complexity(self, val):
        self._complexity = val

    @property
    def pareto_front_rank(self):
        if self._pareto_front_rank is None:
            raise ValueError
        else:
            return self._pareto_front_rank

    @pareto_front_rank.setter
    def pareto_front_rank(self, val):
        assert val >= 1
        self._pareto_front_rank = val

    @property
    def crowding_dist(self):
        if self._crowding_dist is None:
            raise ValueError
        else:
            return self._crowding_dist

    @crowding_dist.setter
    def crowding_dist(self, val):
        assert val >= 0
        self._crowding_dist = val

    @property
    def domination_count(self):
        """Number of solns *this soln is dominated by*"""
        if self._domination_count is None:
            raise ValueError
        else:
            return self._domination_count

    @domination_count.setter
    def domination_count(self, val):
        assert val >= 0
        self._domination_count = val

    @property
    def dominated_set(self):
        """The set of solns that *this soln dominates*"""
        if self._dominated_set is None:
            raise ValueError
        else:
            return self._dominated_set

    @dominated_set.setter
    def dominated_set(self, val):
        self._dominated_set = val

    def does_contain(self, indiv):
        return indiv == self._lv_indiv or indiv == self._rb_indiv

    def __eq__(self, other):
        # use id for equality because may have duplicate indiv genotypes
        # but want to treat them as separate solns
        return id(self) == id(other)
