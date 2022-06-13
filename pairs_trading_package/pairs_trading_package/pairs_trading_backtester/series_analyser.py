import numpy as np
import pandas as pd
import sys
import collections, functools, operator

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, OPTICS, DBSCAN, cluster_optics_dbscan
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

class SeriesAnalyser:
    """
    This class contains a set of functions to deal with time series analysis.
    """

    def __init__(self, seed=0):
        """
        :initial elements
        """
        
        self.seed = seed
        np.random.seed(seed)
        

    def check_for_stationarity(self, X,  subsample=0):
        """
        H_0 in adfuller is unit root exists (non-stationary).
        We must observe significant p-value to convince ourselves that the series is stationary.

        :param X: time series
        :param subsample: boolean indicating whether to subsample series
        :return: adf results
        """
        if subsample != 0:
            subsampled_X = X.resample('B', label='right', closed='right').last().dropna()
            result = adfuller(subsampled_X)
        else:
            result = adfuller(X)
        # result contains:
        # 0: t-statistic
        # 1: p-value
        # others: please see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

        return {'t_statistic': result[0], 'p_value': result[1], 'critical_values': result[4]}
    

    def check_properties(self, train_series: pd.Series, test_series: pd.Series, ex_args: dict):
        """
        Gets two time series as inputs and provides information concerning cointegration stasttics
        Y - b*X : Y is dependent, X is independent
        """

        # perform test manually in both directions
        X = train_series[0]
        Y = train_series[1]
        pairs = [(X, Y), (Y, X)]
        pair_stats = [0] * 2
        
        coint_pval_count = []
        
        criteria_not_verified = 'leg_stationarity'

        # first of all, must verify price series S1 and S2 are I(1)
        stats_Y = self.check_for_stationarity(Y, subsample=ex_args['subsample'])
        if stats_Y['p_value'] > 0.10:
            stats_X = self.check_for_stationarity(X, subsample=ex_args['subsample'])
            if stats_X['p_value'] > 0.10:
                # conditions to test cointegration verified

                for i, pair in enumerate(pairs):
                    S1 = np.asarray(pair[0])
                    S2 = np.asarray(pair[1])
                    S1_c = sm.add_constant(S1)

                    # Y = bX + c
                    # ols: (Y, X)
                    results = sm.OLS(S2, S1_c).fit()
                    b = results.params[1]

                    if b > 0:
                        spread = pair[1] - b * pair[0] # as Pandas Series
                        spread_array = np.asarray(spread) # as array for faster computations

                        criteria_not_verified = 'hurst_exponent'

                        hurst_exponent = self.hurst(spread_array)
                        if hurst_exponent < ex_args['hurst_threshold']:
                            criteria_not_verified = 'half_life'

                            hl = self.calculate_half_life(spread_array)
                            if (hl >= ex_args['min_half_life']) and (hl < ex_args['max_half_life']):
                                criteria_not_verified = 'mean_cross'

                                zero_cross = self.zero_crossings(spread_array)
                                if zero_cross >= ex_args['min_zero_crossings']:
                                    criteria_not_verified = 'cointegration'
                                    
                                    stats = self.check_for_stationarity(spread, subsample=ex_args['subsample'])
                                    
                                    coint_pval_count.append(stats['p_value'])
                                    
                                    if stats['p_value'] < ex_args['p_value_threshold']:  # verifies required pvalue
                                        criteria_not_verified = 'None'

                                        pair_stats[i] = {'t_statistic': stats['t_statistic'],
                                                          'critical_val': stats['critical_values'],
                                                          'p_value': stats['p_value'],
                                                          'coint_coef': b,
                                                          'zero_cross': zero_cross,
                                                          'half_life': int(round(hl)),
                                                          'hurst_exponent': hurst_exponent,
#                                                           'spread': spread,
                                                          'Y_train': pair[1].name,
                                                          'X_train': pair[0].name
                                                          }
                                        break

        if pair_stats[0] == 0 and pair_stats[1] == 0:
            result = None
            return result, criteria_not_verified, coint_pval_count

        elif pair_stats[0] == 0:
            result = 1
        elif pair_stats[1] == 0:
            result = 0
        else: # both combinations are possible
            # select lowest t-statistic as representative test
            if abs(pair_stats[0]['t_statistic']) > abs(pair_stats[1]['t_statistic']):
                result = 0
            else:
                result = 1

        if result == 0:
            result = pair_stats[0]
            result['X_test'] = test_series[0].name
            result['Y_test'] = test_series[1].name
        elif result == 1:
            result = pair_stats[1]
            result['X_test'] = test_series[1].name
            result['Y_test'] = test_series[0].name

        return result, criteria_not_verified, coint_pval_count

    def find_pairs(self, data_train: pd.DataFrame, data_test: pd.DataFrame, ex_args: dict):
        """
        This function receives a df with the different securities as columns, and aims to find tradable
        pairs within this world. There is a df containing the training data and another one containing test data
        Tradable pairs are those that verify:
            - cointegration
            - minimium half life
            - minimium zero crossings

        :param data_train: df with training prices in columns
        :param data_test: df with testing prices in columns
        :param ex_args: (dict)
        :return: pairs that passed test
        """
        n = data_train.shape[1]
        keys = data_train.keys()
        pairs_fail_criteria = {'leg_stationarity': 0,'cointegration': 0, 'hurst_exponent': 0, 'half_life': 0, 'mean_cross': 0, 'None': 0}
        pairs = []
        
        coint_pval_counts = []
        for i in range(n):
            for j in range(i + 1, n):
                S1_train = data_train[keys[i]]; 
                S2_train = data_train[keys[j]]
                
                S1_test = data_test[keys[i]]; 
                S2_test = data_test[keys[j]]
                
                result, criteria_not_verified, coint_pval_count = self.check_properties((S1_train, S2_train), (S1_test, S2_test), ex_args)
                
                coint_pval_counts.append(coint_pval_count)
                pairs_fail_criteria[criteria_not_verified] += 1
                if result is not None:
                    pairs.append((keys[i], keys[j], result))


        return pairs, pairs_fail_criteria, coint_pval_counts

    def calculate_half_life(self, z_array):
        """
        This function calculates the half life parameter of a
        mean reversion series
        """
        z_lag = np.roll(z_array, 1)
        z_lag[0] = 0
        z_ret = z_array - z_lag
        z_ret[0] = 0

        # adds intercept terms to X variable for regression
        z_lag2 = sm.add_constant(z_lag)

        model = sm.OLS(z_ret[1:], z_lag2[1:])
        res = model.fit()

        halflife = -np.log(2) / res.params[1]

        return halflife

    def hurst(self, ts):
        """
        Returns the Hurst Exponent of the time series vector ts.
        Series vector ts should be a price series.
        Source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing"""
        # Create the range of lag values
        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        # Here it calculates the variances, but why it uses
        # standard deviation and then make a root of it?
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0

    def zero_crossings(self, x):
        """
        Function that counts the number of zero crossings of a given signal
        :param x: the signal to be analyzed
        """
        x = x - x.mean()
        zero_crossings = sum(1 for i, _ in enumerate(x) if (i + 1 < len(x)) if ((x[i] * x[i + 1] < 0) or (x[i] == 0)))

        return zero_crossings

    def apply_PCA(self, n_components, df, svd_solver='auto', ignore_first_eigenvector=False, random_state=0):
        """
        This function applies Principal Component Analysis to the df given as
        parameter

        :param n_components: number of principal components
        :param df: dataframe containing time series for analysis
        :param svd_solver: solver for PCA: see PCA documentation
        :return: reduced normalized and transposed df
        """

        if not isinstance(n_components, str):
            if n_components > df.shape[1]:
                print("ERROR: number of components larger than samples...")
                exit()

        pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
        pca.fit(df)
        explained_variance = pca.explained_variance_ratio_

        if ignore_first_eigenvector:
            X = preprocessing.StandardScaler().fit_transform(pca.components_[1:].T)
        else:
            # standardize
            X = preprocessing.StandardScaler().fit_transform(pca.components_.T)

        return X, explained_variance


    def apply_clustering_algo(self, algo, feature_df, set_cols, algo_args):
        """
        
        algo_args expects a dict as follows;
        
        naming_scheme = dict({'kmeans': {'no_clusters': 'n_clusters', 'algo': 'kmeans', 'distance': 'euclidean'},
                              'agglomerative': {'no_clusters': 'n_clusters', 'algo': 'linkage', 'distance': 'affinity'},
                              'spectral': {'no_clusters': 'n_clusters', 'algo': 'spectral', 'distance': 'affinity'},
                              'optics': {'no_clusters': 'min_samples', 'algo': 'optics', 'distance': 'metric'},                      
                              })

        :param algo: (str)
        :param feature_df: (pd.DataFrame)
        :param set_cols: (list)
        :param algo_args: (dict)
        :return: (np.array)
        """

        if algo == 'kmeans':
            clust_algo = KMeans(**algo_args)
            assigned_labels = clust_algo.fit_predict(feature_df)
        elif algo == 'agglomerative':
            clust_algo = AgglomerativeClustering(**algo_args)
            assigned_labels = clust_algo.fit_predict(feature_df)
        elif algo == 'spectral':
            clust_algo = SpectralClustering(**algo_args)
            assigned_labels = clust_algo.fit_predict(feature_df)
        elif algo == 'optics':
            clust_algo = OPTICS(**algo_args)
            assigned_labels = clust_algo.fit_predict(feature_df)
        elif algo == 'dbscan':
            clust_algo = DBSCAN(**algo_args)
            assigned_labels = clust_algo.fit_predict(feature_df)

        n_clusters_ = len(set(assigned_labels))# - (1 if -1 in labels else 0)
        print(algo_args)
        print("Clusters discovered: %d" % n_clusters_)

        clustered_series_all = pd.Series(index=set_cols, data=assigned_labels.flatten())
        clustered_series = clustered_series_all[clustered_series_all != -1]

        counts = clustered_series.value_counts()
        print("Pairs to evaluate: %d" % (counts * (counts - 1) / 2).sum())

        return clustered_series_all, clustered_series, counts, assigned_labels

    def get_candidate_pairs(self, clustered_series, pricing_df_train, pricing_df_test, ex_args: dict):
        """
        This function looks for tradable pairs over the clusters formed previously.

        :param clustered_series: series with cluster label info
        :param pricing_df_train: df with price series from train set
        :param pricing_df_test: df with price series from test set
        :param n_clusters: number of clusters
        :param min_half_life: min half life of a time series to be considered as candidate
        :param min_zero_crosings: min number of zero crossings (or mean crossings)
        :param p_value_threshold: p_value to check during cointegration test
        :param hurst_threshold: max hurst exponent value

        :return: list of pairs and its info
        :return: list of unique tickers identified in the candidate pairs universe
        """
        
        total_pairs, total_pairs_fail_criteria = [], []
        n_clusters = len(clustered_series.value_counts())
        
        coint_pval_counts = []
        
        for clust in range(n_clusters):
            sys.stdout.write("\r"+'Cluster {}/{}'.format(clust+1, n_clusters))
            sys.stdout.flush()
            symbols = list(clustered_series[clustered_series == clust].index)
            cluster_pricing_train = pricing_df_train[symbols]
            cluster_pricing_test = pricing_df_test[symbols]
            
            pairs, pairs_fail_criteria, coint_pval_count = self.find_pairs(cluster_pricing_train,
                                                        cluster_pricing_test, ex_args)
            coint_pval_counts.append(coint_pval_count)
            total_pairs.extend(pairs)
            total_pairs_fail_criteria.append(pairs_fail_criteria)

        print('Found {} pairs'.format(len(total_pairs)))
        unique_tickers = np.unique([(element[0], element[1]) for element in total_pairs])
        print('The pairs contain {} unique tickers'.format(len(unique_tickers)))

        # discarded
        review = dict(functools.reduce(operator.add, map(collections.Counter, total_pairs_fail_criteria)))
        print('Pairs Selection failed stage: ', review)

        return total_pairs, unique_tickers, coint_pval_counts
