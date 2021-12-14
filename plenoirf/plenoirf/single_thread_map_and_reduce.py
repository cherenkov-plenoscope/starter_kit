
def map(func, iterable):
    """
    Retunrs list with results of 'func' for 'iterable'.
    This is a single-thread-dummy for e.g. debugging.
    """
    results = []
    for job in iterable:
        result = func(job)
        results.append(result)
    return results
