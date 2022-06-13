import numpy as np

def abh(pvals, q=0.05):
    """The adaptive linear step-up procedure for controlling the false
    discovery rate.
    Input arguments:
    ================
    pvals - P-values corresponding to a family of hypotheses.
    q     - The desired false discovery rate.
    """
    # P-values equal to 1. will cause a division by zero.
    pvals[pvals>0.99] = 0.99

    # Step 1.
    # If lsu does not reject any hypotheses, stop
    significant = lsu(pvals, q)
    if significant.all() is False:
        return significant

    # Steps 2 & 3
    m = len(pvals)
    sort_ind = np.argsort(pvals)
    m0k = [(m+1-(k+1))/(1-p) for k, p in enumerate(pvals[sort_ind])]
    j = [i for i, k in enumerate(m0k[1:]) if k > m0k[i-1]]

    # Step 4
    mhat0 = int(np.ceil(min(m0k[j[0]+1], m)))

    # Step 5
    qstar = q*m/mhat0
    return lsu(pvals, qstar)

def lsu(pvals, q=0.05):
    """The (non-adaptive) one-stage linear step-up procedure (LSU) for
    controlling the false discovery rate, i.e. the classic FDR method
    proposed by Benjamini & Hochberg (1995).
    Input arguments:
    ================
    pvals - P-values corresponding to a family of hypotheses.
    q     - The desired false discovery rate.
    Output arguments:
    List of booleans indicating which p-values are significant (encoded
    as boolean True values) and which are not (False values).
    """
    m = len(pvals)
    sort_ind = np.argsort(pvals)
    k = [i for i, p in enumerate(pvals[sort_ind]) if p < (i+1.)*q/m]
    significant = np.zeros(m, dtype='bool')
    if k:
        significant[sort_ind[0:k[-1]+1]] = True
    return significant


def benjamini_yekutieli(pvalues, n=None):
    """
    Hochberg's and Hommel's methods are valid when the hypothesis tests are independent or when they are non-negatively
    associated (Sarkar, 1998; Sarkar and Chang, 1997). Hommel's method is more powerful than Hochberg's, but the difference is usually
    small and the Hochberg p-values are faster to compute.
    
    The "BH" and "BY" methods of Benjamini, Hochberg, and Yekutieli control the false discovery rate, the expected proportion of false
    discoveries amongst the rejected hypotheses. The false discovery rate is a less stringent condition than the family-wise error rate,
    so these methods are more powerful than the others.
    
    Note that you can set n larger than length(p) which means the unobserved p-values are assumed to be greater than all the observed p
    for "bonferroni" and "holm" methods and equal to 1 for the other methods.
    
    **References**
    Benjamini, Y., and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. Annals of
    Statistics, 29, 1165–1188. doi: 10.1214/aos/1013699998.
    
    :param pvalues: (np.array) Array of p values to adjust.
    :param n: (int) Number of comparisons, must be at least length(p); only set this (to non-default) when you know what you are doing!
    :return: (np.array) Numpy array of corrected p values (of the same length as p).
    """

    if n is None:
        n = len(pvalues)

    p0 = np.array(pvalues)
    p_a = p0[np.logical_not(np.isnan(p0))]

    lp = len(p_a)

    assert n >= lp

    results = None

    i = np.arange(lp - 1, -1, -1)
    o = (-p_a).argsort()
    ro = np.argsort(o)
    q = np.sum(1 / np.arange(1, n + 1))
    results = np.minimum(1, np.minimum.accumulate((q * n / (i + 1)) * p_a[o]))[ro]

    p0[np.logical_not(np.isnan(p0))] = results

    return p0
