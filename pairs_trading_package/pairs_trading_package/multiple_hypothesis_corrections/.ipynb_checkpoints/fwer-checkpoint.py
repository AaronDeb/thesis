import numpy as np
import pandas as pd

# Adapted from https://github.com/puolival/multipy

def bonferroni(pvals, alpha=0.05):
    """A function for controlling the FWER at some level alpha using the
    classical Bonferroni procedure.
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    return pvals < alpha/float(m)

def hochberg(pvals, alpha=0.05):
    """A function for controlling the FWER using Hochberg's procedure [1].
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    # Sort the p-values into ascending order
    ind = np.argsort(pvals)

    """Here we have k+1 (and not just k) since Python uses zero-based
    indexing."""
    test = [p <= alpha/(m+1-(k+1)) for k, p in enumerate(pvals[ind])]
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind[0:np.sum(test)]] = True
    return significant

def holm_bonferroni(pvals, alpha=0.05):
    """A function for controlling the FWER using the Holm-Bonferroni
    procedure [2].
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    test = [p > alpha/(m+1-k) for k, p in enumerate(pvals[ind])]

    """The minimal index k is m-np.sum(test)+1 and the hypotheses 1, ..., k-1
    are rejected. Hence m-np.sum(test) gives the correct number."""
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind[0:m-np.sum(test)]] = True
    return significant

def sidak(pvals, alpha=0.05):
    """A function for controlling the FWER at some level alpha using the
    procedure by Sidak [3].
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    n, pvals = len(pvals), np.asarray(pvals)
    return pvals < 1. - (1.-alpha) ** (1./n)
