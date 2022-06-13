"""

"""

import pandas as pd
import numpy as np

import itertools
import random

from scipy.stats import norm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import sklearn.metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, OPTICS, DBSCAN
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample, shuffle

#from sklearn_extra.cluster import KMedoids

#from mlfinlab.codependence import (get_dependence_matrix, get_distance_matrix)
#from mlfinlab.data_generation.bootstrap import block_bootstrap

from pairs_trading_package.clustering.infomax import InfoMax
from pairs_trading_package.clustering.clustering_metrics import err_rate, nmi, ari, acc, clustering_accuracy
from pairs_trading_package.utils import flatten


def get_external_clustering_metrics(y_true: np.array, y_pred: np.array) -> tuple:
    """
    
    :param y_true: (np.array)
    :param y_pred: (np.array)
    :return: (tuple)
    """

    acc = np.round(clustering_accuracy(y_true, y_pred), 5)
    nmi_ = np.round(nmi(y_true, y_pred), 5)
    ari_ = np.round(ari(y_true, y_pred), 5)
    
    return acc, nmi_, ari_

def get_internal_clustering_metrics(feature_df: pd.DataFrame, labels: np.array) -> tuple:
    """
    
    :param feature_df: (pd.DataFrame)
    :param labels: (np.array)
    :return: (tuple)
    """

    chs = sklearn.metrics.calinski_harabasz_score(feature_df, labels)

    dbs = sklearn.metrics.davies_bouldin_score(feature_df, labels)

    sil = sklearn.metrics.silhouette_samples(feature_df, labels)
    
    infomax = InfoMax().get_entropies(labels)
    
    return chs, dbs, np.mean(sil)/np.std(sil), infomax[0], infomax[1]

def absolute_error(ytrue, ypred):
    residuals = ytrue - ypred
    return abs(residuals).sum()

def get_label_errors(label_analytics_df, n_clusters):
    """
    
    :param label_analytics_df: (pd.DataFrame)
    :param n_clusters: (int)
    :return: (np.array)
    """
    
    from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error

    error = []
    for i in range(0, n_clusters-1):
        expected_cluster_formation = [np.round(label_analytics_df.iloc[:,i].dropna().sum()/len(label_analytics_df.iloc[:,i].dropna()))]*(len(label_analytics_df.iloc[:,i].dropna()))
        error.append(absolute_error(expected_cluster_formation, label_analytics_df.iloc[:,i].dropna().values))
        
    return error

def run_clustering_algo(algo, feature_df, algo_args):
    """
    
    :param algo: (str)
    :param feature_df: (pd.DataFrame)
    :param algo_args: (dict)
    :return: (np.array)
    """

    if algo == 'kmeans':
        clust_algo = KMeans(**algo_args)
        assigned_labels = clust_algo.fit_predict(feature_df)
    elif algo == 'kmedoids':
        clust_algo = KMedoids(**algo_args).fit(feature_df)
        assigned_labels = clust_algo.predict(feature_df)
    elif algo == 'agglomerative' or algo == 'ward' or algo == 'complete' or algo == 'single' or algo == 'average':
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
        
    return assigned_labels


def permute_params(template):
    """
    
    Will expect template as following;
        {'n_clusters': range(3, 55, 1), 'linkage': ['average']}
    
    :param template: (dict)
    :return: (list)
    """
    
    keys, values = zip(*template.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    return permutations_dicts

def run_clustering_evaluation(feature_df: pd.DataFrame, algo_args: list, ground_truth: np.array = np.array([])) -> tuple:
    """
    
    :param feature_df: (pd.DataFrame)
    :param algo_args: (dict)
    :param ground_truth: (np.array)
    :return: (tuple)
    """

    clust_internal_metrics = []
    clust_external_metrics = []
    label_analytics = []
    label_pair_counts = []

    for algo_s in algo_args.keys():
        
        for algo_arg in algo_args[algo_s]:

            assigned_labels = run_clustering_algo(algo_s, feature_df, algo_arg)

            clust_internal_metrics.append(get_internal_clustering_metrics(feature_df, assigned_labels))

            if (algo_s == 'optics') or (algo_s == 'dbscan'):
                val_counts = pd.Series(assigned_labels)
                val_counts = val_counts[val_counts != -1].value_counts()
            else:
                val_counts = pd.Series(assigned_labels).value_counts()

            label_analytics.append(val_counts)
            label_pair_counts.append((val_counts * (val_counts - 1) / 2).sum())

            if len(ground_truth) != 0:
                clust_external_metrics.append(get_external_clustering_metrics(ground_truth, assigned_labels))
        
    clust_internal_metrics_df = pd.DataFrame(clust_internal_metrics, columns=['C-H', 'D-B', 't-val{Sil}', 'E[s]', 'E[k_s]'])
    clust_external_metrics_df = pd.DataFrame(clust_external_metrics, columns=['Accuracy', 'NMI', 'ARI'])
    
#     label_analytics_df = pd.concat(label_analytics, axis=1)
#     label_analytics_df.columns = range(2, len(label_analytics_df.columns)+2)
    
#     clust_internal_metrics_df['DfU'] = get_label_errors(label_analytics_df, n_clusters)
    clust_internal_metrics_df['pair_count'] = label_pair_counts
    
    return clust_internal_metrics_df, clust_external_metrics_df
    
    
def prepend_as_tuple(list_objs: list, to_prepend: str = ""):
    """
    
    :param list_objs: (list)
    :param to_prepend: (str)
    :return: (list)
    """
    
    final_list = []
    for obj in list_objs:
        if type(obj) == tuple:
            final_list.append(tuple((to_prepend, *obj)))
        else:
            final_list.append(tuple((to_prepend, obj)))
        
    return final_list


def run_dimensionality_evaluation(dimensions_list: list, returns_df: pd.DataFrame, ground_truth: list, algo_args: dict) -> tuple:
    """
    
    :param dimensions_list: (list)
    :param returns_df: (pd.DataFrame)
    :param ground_truth: (list)
    :param algo_args: (dict)
    :return: (tuple)
    """
    
    dim_internal_algo_results = []
    dim_external_algo_results = []

    for feature_dim in dimensions_list:
        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(returns_df)

        pca_instance = PCA(n_components=feature_dim)#, copy_X=False)
        pca_instance.fit(scaled_data)

#         reduced_data = pca_instance.fit_transform(scaled_data)

        feature_df = pd.DataFrame(pca_instance.components_, columns=returns_df.columns).T

        dim_internal_clust_evals = pd.DataFrame()
        dim_external_clust_evals = pd.DataFrame()

        for s_algo in algo_args:
            
            algo_name = list(s_algo.keys())[0]

            clust_internal_metrics_df, clust_external_metrics_df = run_clustering_evaluation(feature_df, algo_args=s_algo, ground_truth=ground_truth)

            clust_internal_metrics_df.columns = prepend_as_tuple(list(clust_internal_metrics_df.columns), algo_name)
            clust_external_metrics_df.columns = prepend_as_tuple(list(clust_external_metrics_df.columns), algo_name)

            dim_internal_clust_evals = pd.concat([dim_internal_clust_evals, clust_internal_metrics_df], axis=1)
            dim_external_clust_evals = pd.concat([dim_external_clust_evals, clust_external_metrics_df], axis=1)

        dim_internal_clust_evals.columns = prepend_as_tuple(list(dim_internal_clust_evals.columns), feature_dim)
        dim_external_clust_evals.columns = prepend_as_tuple(list(dim_external_clust_evals.columns), feature_dim)

        dim_internal_algo_results.append(dim_internal_clust_evals)
        dim_external_algo_results.append(dim_external_clust_evals)
        
    return dim_internal_algo_results, dim_external_algo_results  
       

# def run_clustering_sensitivity_analysis(feature_df: pd.DataFrame, n_simulations: int, ground_truth: list, algo: str, max_clusters: int) -> tuple:
#     """
    
#     :param feature_df: (pd.DataFrame)
#     :param n_simulations: (int)
#     :param ground_truth: (list)
#     :param algo: (str)
#     :param max_clusters: (int)
#     :return: (tuple)
#     """
    
#     bootstrap_dataset = block_bootstrap(feature_df, n_samples=n_simulations, size=feature_df.shape, block_size=(2, 3))
    
#     internal_metric_one = pd.DataFrame()
#     internal_metric_two = pd.DataFrame()
#     internal_metric_three = pd.DataFrame()

#     for n in range(n_simulations):
#         boot_internal_df, boot_external_df = run_clustering_evaluation(bootstrap_dataset[n], ground_truth=ground_truth,
#                                                                        max_clusters=max_clusters, algo=algo)

#         internal_metric_one = pd.concat([internal_metric_one, boot_internal_df['D-B']], axis=1)
#         internal_metric_two = pd.concat([internal_metric_two, boot_internal_df['t-val{Sil}']], axis=1)
#         internal_metric_three = pd.concat([internal_metric_three, boot_internal_df['C-H']], axis=1)

#     return internal_metric_one, internal_metric_two, internal_metric_three



  
def generate_clustering_arg_templates(start_no, end_no, exclude_templates=[]):

    args_templates = dict({'kmeans': {'init':['k-means++'], 'n_clusters': range(start_no, end_no, 4)},
                        'agglomerative': {'n_clusters': range(start_no, end_no, 4), 
                                          'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
                                          'linkage': ['complete', 'average', 'single']}, 
                        'spectral': {'n_clusters': range(start_no, end_no, 4), 'affinity': ['rbf', 'nearest_neighbors'],
                                      'eigen_tol': [1e-5], 'assign_labels': ['discretize']},
                        'optics': {'min_samples': range(3, 5, 1),
                                    'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']}
                        })

    def get_all_permutations(args_template):
        keys, values = zip(*args_template.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    full_arg_templates = dict()
    
    for arg_key in args_templates.keys():
   
        if not arg_key in exclude_templates:
    
            full_arg_templates[arg_key] = get_all_permutations(args_templates[arg_key])

    return full_arg_templates
