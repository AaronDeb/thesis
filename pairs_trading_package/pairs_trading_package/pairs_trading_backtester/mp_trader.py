import sys
import time

import numpy as np
import pandas as pd

from timeit import default_timer as timer

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

from pairs_trading_package.pairs_trading_backtester.trader import Trader
from pairs_trading_package.utils import flatten, postfix_keys_to_dict

class MPTrader(Trader):
    """
    """

    def threshold_strategy_mp_callback(self, args):
        """
        
        :param args: (dict)
        :return: (tuple)
        """

        # YTEST AND XTEST NEED TO BE RENAMED TO SOMETHIGN DIFFERENT
        
        if args['test_mode']:
            y = self.df_prices_test[args['ytest']]
            x = self.df_prices_test[args['xtest']]
        else:
            y = self.df_prices_train[args['ytest']][args['train_val_split']:]
            x = self.df_prices_train[args['xtest']][args['train_val_split']:]

        summary, sharpe, balance_summary = self.threshold_strategy(y=y, x=x,
                                                                     beta=args['coint_coef'],
                                                                     entry_level=args['entry_multiplier'],
                                                                     exit_level=args['exit_multiplier'])

        return (summary, sharpe[1], balance_summary)


    def mp_apply_trading_strategy_with_costs(self, pairs, entry_multiplier=1, exit_multiplier=0,
                                   test_mode=False, train_val_split='2017-01-01'):
        """
        
        :param pairs:
        :param entry_multiplier:
        :param exit_multiplier:
        :param test_mode:
        :param train_val_split:
        :return: (tuple)
        """

        start = timer()

        print(f'starting computations on {cpu_count()} cores')

        pair_identifiers = []
        values = []

        for p in range(len(pairs)):

            pair_info = pairs[p][2]

            pair_identifiers.append((pairs[p][0], pairs[p][1]))

            values.append( dict({'ytest': pair_info['Y_test'],
                                 'xtest': pair_info['X_test'],
                                 'test_mode': test_mode,
                                 'train_val_split': train_val_split,
                                 'coint_coef': pair_info['coint_coef'],
                                 'entry_multiplier': entry_multiplier,
                                 'exit_multiplier': exit_multiplier}) )


        with Pool() as pool:
            res = pool.map(self.threshold_strategy_mp_callback, values)

        cum_returns_with_costs = []
        sharpe_results_with_costs = []
        performance = []

        for pid, r in enumerate(res):
            cum_returns_with_costs.append((r[0].account_balance[-1] - 1) * 100)
            sharpe_results_with_costs.append(r[1])
            performance.append((pairs[pid], r[0], r[2]))

        end = timer()
        print(f'elapsed time: {end - start}')

        return (sharpe_results_with_costs, cum_returns_with_costs), performance
    
