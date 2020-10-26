# Machine Learning - Project 1

The code of the project of team 'answer42' for [EPFL Machine Learning Higgs](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) challenge for year 2020.

## Software requirements

* Python version: 3.6.9

_Disclaimer_: The requirements aren't strict, but are recommended as all the code was tested using them

## Data preprocessing

There is no use of filtering of data samples.

Features are augmented and modified by applying different functions in the following order:

1. Numeric features are standardized
2. Added one hot encoding for the single nominal feature (3 new features created, the initial dropped)
3. Missing columns replaced with the mean of the values in the given column (as all the missing are numeric features, missing columns are set to 0 which is the mean after applying standardization)
4. Sin and cos function applied on all the columns obtained after step 3, the new columns are added beside the already existing ones
5. Top 54 features obtained after step 4 selected (Selection done using backward attribute selection and evaluated using 10-fold cross validation on logistic regression)
6. Features obtained after step 5 are multiplied with each other and the result is added beside the original features after step 5
7. Polynomial degrees 2 and 3 of features obtained after step 5 are added beside the features obtained after step 6
8. For each of the features that had missing values in the starting data added a binary column indicating whether the given value was missing in the data before step 1 and the new columns are added beside the columns obtained after step 7
9. Bias column added beside the features obtained after step 8

## Model

The model is obtained by applying regularized logistic regression on the preprocessed features and trained using mini batch gradient descent.

The model achieves mean accuracy 0.842, F1-score 0.76 on the training set using 5-fold cross validation and mean accuracy 0.84, mean F1 score 0.759 on the test set for the values of training parameters and hyperparameters:

* Trade-off parameter: 10^(-9)
* Learning rate: 0.04
* Batch size: 2000
* Number of epochs: 400

### Author's notes

Pretty late into experimenting with big number of augmented features we noticed that some of the implemented functions (notably cross validation [mainly because of stratification part] and pairwise mutliplication) are not memory optimized for work with huge amount of data and therefore require a lot of memory. This effect is especially noticeable in Google Colab, which was used for testing different setups and models, as sessions would often crash due to the lack of RAM. We don't fully understand the effect this may have on local execution of the code on PC, but expect it to be rather slow if there is not enough RAM. The  files provided to recreate the final submission are expected to require anywhere between 12 and 16 GB of memory.

## Authors

* Andrei Atanov
* Valentina Shumovskaia
* Miloš Vujasinović
