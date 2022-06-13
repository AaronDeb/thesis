"""

"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from munkres import Munkres

from pairs_trading_package.utils import (linear_assignment, shuffle) 

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def best_map(L1, L2):
    """
    """
    
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
            
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index).astype(int)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    dis = [gt_s[i] - c_x[i] for i in range(len(gt_s))]
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate, dis

def clustering_accuracy(labels, predicted_labels):
    """
    
    :param labels: (np.array)
    :param predicted_labels: (np.array)
    :return: (int)
    """
    
    cm = confusion_matrix(labels, predicted_labels)

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    
    return np.trace(cm2) / np.sum(cm2)

def acc(y_true, y_pred):
    """
    
    :param y_true: (np.array) true labels, numpy.array with shape `(n_samples,)`
    :param y_pred: (np.array) predicted labels, numpy.array with shape `(n_samples,)`
    :return: (int) accuracy, in [0,1]
    """
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
