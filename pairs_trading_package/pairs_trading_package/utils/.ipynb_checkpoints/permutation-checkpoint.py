
def shuffle(*arrs):
    """ shuffle.
    Shuffle given arrays at unison, along first axis.
    Arguments:
        *arrs: Each array to shuffle at unison.
    Returns:
        Tuple of shuffled arrays.
    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)
