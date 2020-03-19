## Tutorials
### Tutorial 1 - Binary classification
This tutorial can be found in the [UCI_BreastCancer_BinaryClass.ipynb](https://github.com/apoorva-sh/zodiac-experiments/blob/master/docs/notebooks/UCI_BreastCancer_BinaryClass.ipynb) notebook.

In this notebook we use the [UCI Breast cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset and run classification using sklearn's SVM and RandomForest.

Metric plots for accuracy, recall, f1, and precision are generated for both of these models both for manual grids and parzen windows

### Tutorial 2 - Multiclass classification
This tutorial can be found in the [MNIST_MultiClass.ipynb](https://github.com/apoorva-sh/zodiac-experiments/blob/master/docs/notebooks/MNIST_MultiClass.ipynb) notebook.

In this notebook we use the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset and run classification using sklearn's RandomForest and Naive Bayes.

Metric plots for accuracy, recall, f1, and precision are generated for both of these models both for manual grids and parzen windows

## Running these notebooks

- To run the tutorial notebooks clone this repository 
- Launch jupyter notebook in "zodiac-experiments" and open the tutorial you wish to run and run all cells
- To run the MultiClass classification notebook, please download the csv for test and train data set from [here](https://pjreddie.com/projects/mnist-in-csv/) and store it in a folder named "data" in "zodiac-experiments"
