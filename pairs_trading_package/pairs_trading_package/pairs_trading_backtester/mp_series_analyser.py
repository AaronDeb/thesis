import sys
import time

import numpy as np
import pandas as pd

from timeit import default_timer as timer
from datetime import timedelta

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

from pairs_trading_package.pairs_trading_backtester.series_analyser import SeriesAnalyser
from pairs_trading_package.utils import flatten, postfix_keys_to_dict

class MPSeriesAnalyser(SeriesAnalyser):
    """
    """
    

    def check_if_cached(self, pair_results_cache, pair_0, pair_1):
        """
        Goes through the cache that stores previously calculated pair relationships and
        returns data specific to the relationship if found.
        
        :param pair_results_cache: (list)
        :param pair_0: (str)
        :param pair_1: (str)
        :return: (bool)
        """
        
        for c in pair_results_cache:
            if (c[0] == pair_0 and c[1] == pair_1) or (c[0] == pair_1 and c[1] == pair_0):
                return c

        return False

    def check_properties_mp_callback(self, args):
        """
        Wrapper that is obligatory for the multi processing code. 
        It basically prepares training and test series for processing.
        
        :param args: (dict)
        """
        
        key_i = args['key_i']
        key_j = args['key_j']

        S1_train = self.df_prices_train[key_i]; 
        S2_train = self.df_prices_train[key_j]
        S1_test = self.df_prices_test[key_i]; 
        S2_test = self.df_prices_test[key_j]

        result, criteria_not_verified, coint_pval_count = self.check_properties((S1_train, S2_train), 
                                                                                (S1_test, S2_test), args['ex_args'])

        return (result, criteria_not_verified, coint_pval_count)


    def mp_apply_check_properties(self, pair_results_cache, clustered_series, ex_args):

        start = timer()

        print(f'starting computations on {cpu_count()} cores')

        working_args_list = []
        already_processed_pairs = []
        n_clusters = len(clustered_series.value_counts())

        for clust in range(n_clusters):
            
            symbols = list(clustered_series[clustered_series == clust].index)

            for i in range(len(symbols)):
                
                for j in range(i + 1, len(symbols)):

                    if self.check_if_cached(pair_results_cache, symbols[i], symbols[j]) == False:

                        working_args = dict({'key_i': symbols[i], 'key_j': symbols[j], 'ex_args': ex_args})
                        
                        working_args_list.append(working_args)

                    else:

                        already_processed_pairs.append((symbols[i], symbols[j]))

        with Pool() as pool:
            res = pool.map(self.check_properties_mp_callback, working_args_list)

        ###############################################################################

        pairs_fail_criteria = {'leg_stationarity': 0, 'cointegration': 0, 
                               'hurst_exponent': 0, 'half_life': 0, 'mean_cross': 0, 'None': 0}
        pairs = []

        coint_pval_counts = []

        for pid, r in enumerate(res):

            # 0,         1,                  2,                 
            #result, criteria_not_verified, coint_pval_count

            coint_pval_counts.append(r[2])
            pairs_fail_criteria[r[1]] += 1
            pairs.append((working_args_list[pid]['key_i'], working_args_list[pid]['key_j'], r[0]))

        for ap in already_processed_pairs:

            pair_val = self.check_if_cached(pair_results_cache, ap[0], ap[1])
            r = pair_val[2]
            if r != None:
                coint_pval_counts.append([r['p_value']])
                pairs.append((ap[0], ap[1], r))


        end = timer()
        print(f'elapsed time: {end - start}')

        ###############################################################################

        print('Found {} pairs'.format(len(pairs)))
        unique_tickers = np.unique([(element[0], element[1]) for element in pairs])
        print('The pairs contain {} unique tickers'.format(len(unique_tickers)))

        # discarded
        print('Pairs Selection failed stage: ', pairs_fail_criteria)

        return pairs, pairs_fail_criteria, coint_pval_counts
