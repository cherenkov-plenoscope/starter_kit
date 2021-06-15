
def map(func, iterable):
    results = []
    for job in iterable:
        result = func(job)
        results.append(result)
    return results
