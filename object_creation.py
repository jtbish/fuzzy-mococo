_curr_indiv_id = 0
_curr_soln_id = 0


def get_next_indiv_id():
    global _curr_indiv_id
    _curr_indiv_id += 1
    return _curr_indiv_id


def get_next_soln_id():
    global _curr_soln_id
    _curr_soln_id += 1
    return _curr_soln_id
