# What is What

## Folders
- ./logs/\*/\* - Hosts HParam logs of parameter searches and all the saved models.
- ./ae_exhaust/embedding_sets/\*/\*.csv - Here are Standard Scaled Feature Vectors from the various AE models that were trained. These are to be used for the clustering stage.
- ./ae_exhaust/saved_models/\*/\*.h5 - Here are the whole model as H5 files that can be loaded as; new_model = tf.keras.models.load_model('my_model.h5')

The names from the embedding_sets and saved_models folders follow the same convention; the folder is the identifier of the period used for training and the name of the file is the same between folders.

## Notebooks

- ae_vs_pca.ipynb - Has experiments involving naive comparisons between Deep/Shallow/Linear Ae and PCA. (TO BE DELETED)
- autoencoder_testbed.ipynb - Original first try at development of AutoEncoders. (TO BE DELETED)
- basic_autoencoder.ipynb - Rudimentary parameter search for Autoencoders and basic RMT based model diagnostics (TO BE DELETED)
- basic_clustering_autoencoder_single.ipynb - Current working implementation of the AutoEncoder as a single object without infrastructure for hyper param automation.
- basic_clustering_autoencoder-hparam.ipynb - Current working implementation of the AutoEncoder with HParam code wrapped around.
- basic_clustering_autoencoder.ipynb - V2 Old Backup (TO BE DELETED)
- basic_clustering_gcn.ipynb - Decent try at GCN (TO BE MOVED)
- deep_filtering_data_generation.ipynb - Needs to be placed somewhere more appropriate. (TO BE MOVED)
- weight_watcher_autoencoder_data_gen.ipynb - Some more RMT model diagnostics. (TO BE DELETED)

## Data Files

- simple_nn_concat_dfs.csv - Summary Model Diagnostics from WW. 
- nn_concat_dfs.csv - Detailed Model Diagnostics from WW.

- 354784217180c7efa1864de1d4a0a80f_SPLIT_IDX_2_ae.csv - AutoEncoder Results using v1 platform.
- 4a8ad82d19333a18d2c6dd0fbd19fce7_SPLIT_IDX_1_ae.csv - AutoEncoder Results using v1 platform.
- 7ce5583986b4adea2f6ba98ce466d520_SPLIT_IDX_0_ae.csv - AutoEncoder Results using v1 platform.