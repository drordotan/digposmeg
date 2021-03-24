

n_positions = 6

unit_digits = 2, 3, 5, 8
decade_digits = 2, 3, 4, 5, 8

#--------------------------------------------------------------------
def all_targets(ndigits=None, include_4=True):

    assert ndigits in (1, 2, None)

    targets = []

    if ndigits is None or ndigits == 1:
        targets.extend(unit_digits)

    if ndigits is None or ndigits == 2:
        decades = decade_digits if include_4 else unit_digits
        targets.extend([d * 10 + u for d in decades for u in (2, 3, 5, 8)])

    return targets

#--------------------------------------------------------------------
def all_event_ids(ndigits=None, include_4=True):

    result = []
    if ndigits is None or ndigits == 1:
        result.extend([target * 10 + pos for target in all_targets(ndigits=1, include_4=include_4) for pos in range(6)])

    if ndigits is None or ndigits == 2:
        result.extend([target * 10 + pos for target in all_targets(ndigits=2, include_4=include_4) for pos in range(5)])

    return result


#--------------------------------------------------------------------
def all_2digit_stimuli():
    targets = all_targets(2, include_4=True)
    return [dict(target=t, location=pos) for t in targets for pos in range(n_positions-1)]


#--------------------------------------------------------------------
def all_conditions(ndigits=None, include_4=True):

    conds = []

    if ndigits is None or ndigits == 1:
        conds += tuple([(target, pos) for target in all_targets(1, include_4) for pos in range(n_positions)])

    if ndigits is None or ndigits == 2:
        conds += tuple([(target, pos) for target in all_targets(2, include_4) for pos in range(n_positions - 1)])

    return conds

#--------------------------------------------------------------------
def ndigits(n):
    return len(str(n))


#--------------------------------------------------------------------
def as_digits(n):
    return [int(d) for d in str(n)]

