import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

from pairs_trading_package.multiple_hypothesis_corrections.fdr import *
from pairs_trading_package.multiple_hypothesis_corrections.fwer import *

def invert_bool(int_seq):
    return (~(np.array(int_seq).astype(bool))).astype(int)

def correct_fwer(method, pvalue, alpha=0.05):
    
    series_pval = pd.Series(pvalue).sort_values()
    
    pvals = series_pval.values

    significant_pvals = method(pvals, alpha=alpha)

    sortedback_pvals = pd.Series(data=significant_pvals, index=series_pval.index).sort_index().values*1.0
    
    return sortedback_pvals


def correct_fdr(method, pvalue, q=0.05):
    series_pval = pd.Series(pvalue).sort_values()
    
    pvals = series_pval.values

    significant_pvals = method(pvals, q=q)

    sortedback_pvals = pd.Series(data=significant_pvals, index=series_pval.index).sort_index().values*1.0
    
    return sortedback_pvals


def get_scores(ytrue, ypred):
    
    cm = confusion_matrix(ytrue, ypred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn)

    fdr = fp / (fp + tp)

    tpr = tp / (tp + fn)
    
    tnr = tn / (tn + fp)
    
    odds_ratio = np.log( (tp * tn) / (fn * fp) )

    f1 = f1_score(ytrue, ypred)
    
    weighted_f1 = f1_score(ytrue, ypred, average='weighted')
    
    fwer = 1 if fp != 0 else 0
    
    return dict({'fpr': fpr, 'fwer': fwer, 'fdr': fdr, 'tpr': tpr, 'tnr': tnr, 'f1': f1, 'weighted_f1': weighted_f1}) 


def get_mhc_panel(select_pvalue, select_cointegratedActual, alpha=0.05, independent_experiments = 2):

    holm_bonferroni_scores = get_scores(select_cointegratedActual*1.0, invert_bool(correct_fwer(holm_bonferroni, select_pvalue)))
                                        #holm_bonf_fwer(alpha, experiments))
    bonferroni_scores = get_scores(select_cointegratedActual*1.0, invert_bool(correct_fwer(bonferroni, select_pvalue)),
                                   alpha/independent_experiments)
    sidak_scores = get_scores(select_cointegratedActual*1.0, invert_bool(correct_fwer(sidak, select_pvalue)),
                              1-(1-alpha)**(1/independent_experiments))

    by_scores = get_scores(select_cointegratedActual*1.0, invert_bool( (benjamini_yekutieli(select_pvalue) < alpha).astype(int) ), 
                           1 - (1 - alpha)**independent_experiments)
    abh_scores = get_scores(select_cointegratedActual*1.0, invert_bool(correct_fdr(abh, select_pvalue)), 1 - (1 - alpha)**independent_experiments)
#     tst_scores = get_scores(select_cointegratedActual*1.0, invert_bool(correct_fdr(tst, select_pvalue)), 1 - (1 - alpha)**independent_experiments)
    np_scores = get_scores(select_cointegratedActual*1.0, (~(select_pvalue<0.05)).astype(int), 1 - (1 - alpha)**independent_experiments)

    scores_list = [holm_bonferroni_scores.values(), bonferroni_scores.values(),
                   sidak_scores.values(), by_scores.values(), abh_scores.values(), #tst_scores.values(), 'ben-hoch',
                   np_scores.values()]

    scores_df = pd.DataFrame(data=scores_list, columns=np_scores.keys(), index=['holm-bonf', 'bonf', 'sid', 'ben-yek', 'abh', 'ney-per'])

    return scores_df
