
import pandas as pd
import numpy as np

from random import randint

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import get_path_collection_extents

from localfinlab.utils import flatten

def get_n_colors(n: int) -> list:
    """
    
    :param n: (int) N number of colors.
    :return: (list) List of Hex color codes.
    """
    
    color = []
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    return color


def getbb(sc, ax):
    """ 
    Function to return a list of bounding
    boxes in data coordinates for a scatter plot 
    
    :param sc:
    :param ax:
    :return:        
    """
    ax.figure.canvas.draw() # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]]*len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]]*len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t],
                [o], transOffset.frozen())
            bboxes.append(result.inverse_transformed(ax.transData))

    return bboxes


def plot_clusterings(flat_dim_data_df: pd.DataFrame, etfs_unique: list, clust_labels_: np.array):
    """
    
    :param flat_dim_data_df: (pd.DataFrame)
    :param etfs_unique: (list)
    :param clust_labels_: (np.array)
    :return: (None)    
    """

    unique_sectors = np.unique(etfs_unique['Sector'].dropna().values)
    sect_colors = get_n_colors(len(unique_sectors))

    no_of_classes = len(np.unique(clust_labels_))

    fig = plt.figure(facecolor='white', figsize=(15, 10))

    ax_object = fig.add_subplot(111)

    # Set spine styling.
    ax_object.spines['left'].set_position('center')
    ax_object.spines['left'].set_alpha(0.3)
    ax_object.spines['bottom'].set_position('center')
    ax_object.spines['bottom'].set_alpha(0.3)
    ax_object.spines['right'].set_color('none')
    ax_object.spines['top'].set_color('none')

    ax_object.xaxis.set_ticks_position('bottom')
    ax_object.yaxis.set_ticks_position('left')
    ax_object.tick_params(which='major', labelsize=18)
    
    ax_object.set_xlim((min(flat_dim_data_df.iloc[:, 0])*1.1, max(flat_dim_data_df.iloc[:, 0])*1.1))
    ax_object.set_ylim((min(flat_dim_data_df.iloc[:, 1])*1.1, max(flat_dim_data_df.iloc[:, 1])*1.1))

    paths_collection = []

    cluster_collection = []

    # For each cluster.
    for cluster in range(0, no_of_classes):

        # Get specific cluster data from the tsne process dataframe.
        cluster_data = flat_dim_data_df[clust_labels_ == cluster]

        if len(cluster_data) > 0:
            # Plot the cluster data by column index [0, 1] -> [x, y].
            paths = ax_object.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1],
                                   s=30**2,
                                   alpha=0.75, marker='.', linestyle='None')
            
            boxes = getbb(paths, ax_object)

            for i, node_name in enumerate(cluster_data.index):
                ax_object.annotate(node_name, (boxes[i].x1, boxes[i].y1))

                node_attr = etfs_unique[etfs_unique['Ticker'].isin([node_name])]['Sector'].values

                if type(node_attr[0]) == str:
                    attr_index = list(unique_sectors).index(node_attr)
                                        
                    rect = plt.Rectangle((boxes[i].x0, boxes[i].y0),
                                         boxes[i].width, boxes[i].height,
                                         fill=False, edgecolor=sect_colors[attr_index], lw=2)
                    
                    ax_object.add_patch(rect)

    attr_patches = []

    for colr_id, attr_col in enumerate(sect_colors):
        attr_patches.append(mpatches.Patch(color=attr_col, label=unique_sectors[colr_id]))

    ax_object.legend(handles=attr_patches)
    
    

def plot_internal_clustering_metrics(dim_internal_algo_results, file_info=None, savefig=False, highlight_excessive_pair_counts=False):
    """
    
    :param dim_internal_algo_results: (pd.DataFrame)
    """
    
    tcolors = get_n_colors(20)
    
    titles = ['Calinski-Harabasz index (Optimal Value - Max) ~ Data Period ' + str(file_info['period']),
              'Davies-Boulding index (Optimal Value - Min) ~ Data Period ' + str(file_info['period']),
              'T-Val of Silhouette scores (Optimal Value - Max) ~ Data Period ' + str(file_info['period']),
              'E[s] ~ Data Period ' + str(file_info['period']),
              'E[k_s] ~ Data Period ' + str(file_info['period']),
              'Unique Pair Combinations Counts ~ Data Period ' + str(file_info['period'])]
    
    y_units = ['C-H Index', 'D-B Index', 'Silhoutte Score', 'Nats', 'Nats', 'Pair Count']

    for color_idx, scoring_fn_str in enumerate(['C-H', 'D-B', 't-val{Sil}', 'E[s]','E[k_s]', 'pair_count']):
        plt.figure(figsize=(10, 5))
        cols_to_show = []
        collection_to_average = pd.DataFrame()

        for dim_idx in range(0, len(dim_internal_algo_results)):
            working_algo_results = dim_internal_algo_results[dim_idx]
            results_cols = working_algo_results.columns
            
            cols_without_unscaled_cols = np.array(list(scoring_fn_str in d for d in results_cols))
            specific_dim_scoringfn_data = working_algo_results.loc[:, results_cols[cols_without_unscaled_cols]]
            
            if len(specific_dim_scoringfn_data) != 0:
                if highlight_excessive_pair_counts == False:
                    plt.plot(specific_dim_scoringfn_data)
                else:
                    pc_algo_results = dim_internal_algo_results[0]

                    mask_cols_get_paircounts = np.array(list('pair_count' in d for d in results_cols))
                    paircount_per_unit_score = pc_algo_results.loc[:, results_cols[mask_cols_get_paircounts]]
                    
                    greater_than_partition = paircount_per_unit_score > 500
                    less_than_partition = paircount_per_unit_score <= 500
                    
                    if scoring_fn_str == 'C-H':
                        plt.plot(np.log(specific_dim_scoringfn_data[greater_than_partition.values]), alpha=0.5)
                        plt.plot(np.log(specific_dim_scoringfn_data[less_than_partition.values]))
                    else:
                        plt.plot(specific_dim_scoringfn_data[greater_than_partition.values], alpha=0.5)
                        plt.plot(specific_dim_scoringfn_data[less_than_partition.values])
#             else:
#                 plt.plot(specific_dim_scoringfn_data, color=tcolors[dim_idx]);
                
            collection_to_average = pd.concat([collection_to_average, specific_dim_scoringfn_data], axis=1)
            cols_to_show.append( results_cols[cols_without_unscaled_cols] )

#         plt.plot(collection_to_average.mean(axis=1).values, color='red', linestyle='--')
        plt.xlabel('Number of Clusters');
        plt.ylabel(y_units[color_idx]);
        plt.title(titles[color_idx]);

        plt.legend(flatten(cols_to_show));
        
        if savefig == True:
            plt.savefig("./images/internal_clust_metrics_" + str(scoring_fn_str) +
                        "_pca_" + str(file_info['pca_comp']) + "_period_" + str(file_info['period']) + ".png")
        
        plt.show()
        
# def plot_internal_clustering_metrics(dim_internal_algo_results, file_info=None, savefig=False):
#     """
    
#     :param dim_internal_algo_results: (pd.DataFrame)
#     """
    
#     tcolors = get_n_colors(20)
    
#     titles = ['Calinski-Harabasz index (Optimal Value - Max) ~ Data Period ' + str(file_info['period']),
#               'Davies-Boulding index (Optimal Value - Min) ~ Data Period ' + str(file_info['period']),
#               'T-Val of Silhouette scores (Optimal Value - Max) ~ Data Period ' + str(file_info['period']),
#               'InfoMax ~ Data Period ' + str(file_info['period']),
#               'Unique Pair Combinations Counts ~ Data Period ' + str(file_info['period'])]
    
#     y_units = ['C-H Index', 'D-B Index', 'Silhoutte Score', 'Normalized Entropy', 'Pair Count']

#     for color_idx, scoring_fn in enumerate(['C-H', 'D-B', 't-val{Sil}', 'infomax', 'pair_count']):
#         plt.figure(figsize=(10, 5))
#         cols_to_show = []
#         collection_to_average = pd.DataFrame()

#         for dim_idx in range(0, len(dim_internal_algo_results)):
#             cols_without_unscaled_cols = np.array(list(scoring_fn in d for d in dim_internal_algo_results[dim_idx].columns))
#             specific_dim_scoringfn_data = dim_internal_algo_results[dim_idx].loc[:, dim_internal_algo_results[dim_idx].columns[cols_without_unscaled_cols]]
            
#             if len(dim_internal_algo_results) == 1:
#                 plt.plot(specific_dim_scoringfn_data)
#             else:
#                 plt.plot(specific_dim_scoringfn_data, color=tcolors[dim_idx]);
                
#             collection_to_average = pd.concat([collection_to_average, specific_dim_scoringfn_data], axis=1)
#             cols_to_show.append( dim_internal_algo_results[dim_idx].columns[cols_without_unscaled_cols] )

# #         plt.plot(collection_to_average.mean(axis=1).values, color='red', linestyle='--')
#         plt.xlabel('Number of Clusters');
#         plt.ylabel(y_units[color_idx]);
#         plt.title(titles[color_idx]);

#         plt.legend(flatten(cols_to_show));
        
#         if savefig == True:
#             plt.savefig("./images/internal_clust_metrics_" + str(scoring_fn) +
#                         "_pca_" + str(file_info['pca_comp']) + "_period_" + str(file_info['period']) + ".png")
        
#         plt.show()
        
def plot_external_clustering_metrics(dim_external_algo_results):
    """
    
    :param dim_external_algo_results: (pd.DataFrame)
    """
    
    tcolors = get_n_colors(20)
    
    titles = ['Cluster Accuracy (Optimal Value - Max = 1)',
              'Normalized Mutual Information (Optimal Value - Max = 1)',
              'Adjusted Rand Index (Optimal Value - Max = 1)']

    for color_idx, scoring_fn in enumerate(['Accuracy', 'NMI', 'ARI']):
        plt.figure(figsize=(10, 5))
        cols_to_show = []
        collection_to_average = pd.DataFrame()

        for dim_idx in range(0, len(dim_external_algo_results)):
            cols_without_unscaled_cols = np.array(list(scoring_fn in d for d in dim_external_algo_results[dim_idx].columns))
            specific_dim_scoringfn_data = dim_external_algo_results[dim_idx].loc[:, dim_external_algo_results[dim_idx].columns[cols_without_unscaled_cols]]
            
            if len(dim_external_algo_results) == 1:
                plt.plot(specific_dim_scoringfn_data)
            else:
                plt.plot(specific_dim_scoringfn_data, color=tcolors[dim_idx]);

            collection_to_average = pd.concat([collection_to_average, specific_dim_scoringfn_data], axis=1)
            cols_to_show.append( dim_external_algo_results[dim_idx].columns[cols_without_unscaled_cols] )

        plt.plot(collection_to_average.mean(axis=1).values, color='red', linestyle='--')
        plt.xlabel('Number of Clusters');
        plt.ylabel('Scoring function units');
        plt.title(titles[color_idx]);

        plt.legend(flatten(cols_to_show));
        plt.show()
        

def plot_stacked_bar_chart(clustered_series, etfs_unique, legend=True):
    """
    
    :param clustered_series: (pd.Series)
    :param etfs_unique: (pd.Series)
    :param legend: (bool)
    :return: (None)
    """
    
    n_clusters = max(clustered_series)+1

    unique_sectors = np.unique(etfs_unique['Sector'].dropna().values)

    sector_based_separation_dict = dict((sect, 0) for sect in unique_sectors)

    clust_based_separation_dict = dict( (str(clust), sector_based_separation_dict.copy()) for clust in range(n_clusters))

    for usect in unique_sectors:
        tickers_in_sector = etfs_unique[etfs_unique['Sector'] == usect]['Ticker']
        for stick in tickers_in_sector:
            if stick in clustered_series:
                clust_based_separation_dict[str(clustered_series[stick])][usect] += 1
                
                
    fig, ax = plt.subplots()

    prev_count = np.zeros(n_clusters)

    for sect in unique_sectors:

        clust_sect_counts = [clst[1][sect] for clst_idx, clst in enumerate(clust_based_separation_dict.items())]

        if len(prev_count) == 0:
            ax.bar(clust_based_separation_dict.keys(), clust_sect_counts, 0.35, label=sect)
        else:
            ax.bar(clust_based_separation_dict.keys(), clust_sect_counts, 0.35, bottom=prev_count, label=sect)

        prev_count = prev_count + np.array(clust_sect_counts)

    if legend:
        ax.legend()

    ax.set_ylabel('No of Assets')
    ax.set_xlabel('Cluster Index')
    ax.set_title('Sector Composition of Individual Clusters')
    plt.show()