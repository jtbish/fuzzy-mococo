from util import get_possibly_null_attr
from multi_objective import MIN_PARETO_FRONT_RANK, MIN_CROWDING_DIST
from object_creation import get_next_indiv_id


class Indiv:
    """Data type to represent an individual in either population."""
    def __init__(self, subspecies_tag, genotype):
        self._subspecies_tag = subspecies_tag
        self._genotype = genotype
        self._phenotype = None
        self._pareto_front_rank = None
        self._crowding_dist = None
        self._indiv_id = get_next_indiv_id()

    @property
    def subspecies_tag(self):
        return self._subspecies_tag

    @property
    def genotype(self):
        return self._genotype

    @property
    def phenotype(self):
        return get_possibly_null_attr(self, "_phenotype")

    @phenotype.setter
    def phenotype(self, val):
        self._phenotype = val

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

    def __eq__(self, other):
        # use global indiv id for equality because may have duplicate genotypes
        # generated but want each new genotype created to be its own indiv
        return self._indiv_id == other._indiv_id
