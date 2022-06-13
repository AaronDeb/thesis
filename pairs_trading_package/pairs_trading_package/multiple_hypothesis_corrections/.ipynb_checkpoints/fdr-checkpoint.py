# Adapted from https://github.com/puolival/multipy

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