import numpy as np
from collections import Counter

class InfoMax:
    """
    Cubero, R.J., Jo, J., Marsili, M., Roudi, Y. and Song, J., 2019. Statistical criticality arises in most informative representations. Journal of Statistical Mechanics: Theory and Experiment, 2019(6), p.063402.
    Sikdar, S., Mukherjee, A. and Marsili, M., 2020. Unsupervised ranking of clustering algorithms by INFOMAX. Plos one, 15(10), p.e0239331.
    
    """

    def safe_ln(self, x):
        if x <= 0:
            return 0
        return np.log(x)

    def get_entropies(self, labels, normalized=True):
        # Ks be the number of times s was observed in the sample
        ks_counts = np.asarray(Counter(labels).most_common()) 

        s_under = len(ks_counts) 

        n = len(labels)
        # -np.sum([(ks_counts[s, 1]/n) * np.log2(ks_counts[s, 1]/n) for s in range(1, to_s)])
        h_of_s = lambda to_s : -np.sum([((k*m_k(k))/n) * self.safe_ln(k/n) for k in range(1, to_s)])

        # m_k is the number of states s that are sampled exactly k times
        m_k = lambda k : np.sum(ks_counts[:,1].astype(int) == k)

        #h_of_s(to_k)
        h_of_k = lambda to_k : - np.sum([ ((k*m_k(k))/n) * self.safe_ln((k*m_k(k))/n) for k in range(1, to_k) ])

        if normalized:
            return h_of_s(s_under)/self.safe_ln(n), h_of_k(s_under)/self.safe_ln(n)
        else:
            return h_of_s(s_under), h_of_k(s_under)

