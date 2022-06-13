# Readme

## Data

Download the data folder from; https://drive.google.com/file/d/1kR5Zy9u87HIFNxx50o1faHHa5S44rJo_/view?usp=sharing

## To execute any notebook you need to install the pairs_trading_package

pip install -e ./pairs_trading_package

## Experiments

### Experiment 1
- clustering_main.ipynb - Notebook that is used to generate the clustering evaluation metrics.
- thesis_backtest_gen.ipynb - Notebook that is used to generate the financial trading metrics.

### Experiment 2
- regime_feature_generator.ipynb - Notebook that is used to generate the features to be appended to the performance dataset.
- experiment_2.ipynb - Notebook that is used for the SHAP analysis of each regression model.

### Experiment 3

- basic_clustering_autoencoder-hparam.ipynb - Implementation of the AutoEncoder with HParam code wrapped around.
- results_generator.ipynb - Notebook used to generate pretty results csv files.
- thesis_backtest_gen-ae.ipynb - Notebook that is used to generate the financial trading metrics.