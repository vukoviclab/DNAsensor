# DNAsensor
___
DNAsensor codes train and test classification/regression supervised learning models for DNA sequences. For the regression models, 
the codes use the Support Vector Machine algorithm from the scikit learn package and for the classification models the codes use  
Convolutional neural network in 2 dimensions from the Tensorflow package. The detailed results are published here: [https://www.biorxiv.org/content/10.1101/2021.08.20.457145v1](https://www.biorxiv.org/content/10.1101/2021.08.20.457145v1)

The scripts are compatible with Python 3, Tensorflow 2 and Scikit-learn 0.24.

## Script files description:

* **ml_data_preparation.py** </br>
    one-hot encoding the data and make a one dimensional vector

* **nn_data_preparation.py** </br>
    one-hot encoding the data and make a two-dimensional vector

* **ml_regression.py** </br>
    run the three different SVR algorithms (Linear, RBF and Sigmoid)

* **nn_classification.py** </br>
    Building the models and then run the Conv2D models
