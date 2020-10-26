import gc
import time
import numpy as np

import data_io
import data_preprocessing
import validation
import attribute_selection
import evaluators
import metrics
from implementations import *

# Constants
DATA_FILE_PREFIX = 'data/'

# Loading training data
y_train, x_train, _, cols = data_io.load_csv_data(f'{DATA_FILE_PREFIX}train.csv')

# Creating column name to index maping
col_to_index_mapping = {col_name: index - 2 for index, col_name in enumerate(cols) if index >= 2}

# Transforming labels from {-1, 1} to {0, 1}
y_train = (y_train + 1) // 2

# Function for data preprocessing
def transformation_pipeline(x, col_to_index_mapping=col_to_index_mapping, transformation_memory=None):
    # Memory is required in order to apply same transformation on training and test data
    training = transformation_memory is None
    if training:
      transformation_memory = {}

    tx = np.copy(x) # Recommended to copy x so it doesn't change

    # Creating binary column indicating whether given column is missing for 
    #   each column that contains missing values
    if training:
      columns_with_missing_values = np.max((tx == -999), axis=0)
      transformation_memory['columns_with_missing_values'] = columns_with_missing_values
    missing_columns_binary = (tx[:, transformation_memory['columns_with_missing_values']] == -999)\
              .astype(int)
    
    # remove missing values with NANs
    tx[tx == -999.] = np.nan

    # Calculate mean and standard deviation in order to to later standardize data
    base_standardize_col_idx = [col_to_index_mapping[key] for key in col_to_index_mapping if 'PRI_jet_num' not in key]
    base_standardize_cols = tx[:, base_standardize_col_idx]
    if training:
      mean = np.nanmean(base_standardize_cols, axis=0)
      stddev = np.nanstd(base_standardize_cols, axis=0)
      transformation_memory['base_mean'] = mean
      transformation_memory['base_stddev'] = stddev

    # Standardize data
    tx[:, base_standardize_col_idx] = (base_standardize_cols - transformation_memory['base_mean']) \
          / transformation_memory['base_stddev']

    # standardize and normalize may change value of fields from default missing values, so it uses matrix calculated before 
    #   applying transformations (0 = mean after standardization)
    tx[np.isnan(tx)] = 0
     
    # onehot for categorical and drop one level
    tx, col_to_index_mapping_upd = data_preprocessing.one_hot_transformation(tx, 'PRI_jet_num', col_to_index_mapping)
    tx = tx[:, :-1]

    # Augment features using sin and cos
    sins = np.sin(tx)
    coses = np.cos(tx)
    tx = np.concatenate((tx, sins, coses), axis=1)
    
    # Select best features (determined using backwards attribute selection)
    first_selection_attr = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 19, 21, 24, 29, 30, 32, 33, 34, 35, 39, 40, 41, 43, 44, 46, 48, 49, 51, 56, 57, 58, 61, 62, 64, 65, 66, 67, 68, 71, 73, 74, 75, 78, 80, 81, 86, 87, 90, 93, 94, 95]
    tx = tx[:, first_selection_attr]
    
    d = tx.shape[1]
    
    # Add polynomial degrees 2 and 3 for the selected features
    poly = data_preprocessing.build_poly(tx, list(range(d)), [2, 3])

    if training:
      poly_mean = np.nanmean(poly, axis=0)
      poly_stddev = np.nanstd(poly, axis=0)
      transformation_memory['poly_mean'] = poly_mean
      transformation_memory['poly_stddev'] = poly_stddev

    # Standardize value of polynomial degrees
    poly = (poly - transformation_memory['poly_mean']) / transformation_memory['poly_stddev']

    # Add features multiplied with each other, stratified polynomial degrees 2 and 3 and
    #   binary columns for missing and clipped features
    tx = np.c_[
               data_preprocessing.build_pairwise_alt(tx, list(range(d))), 
               poly, missing_columns_binary]

    # Add bias
    tx = data_preprocessing.prepend_bias_column(tx)
    
    return tx, transformation_memory


# Makes prediction function out of given parameters
def make_predictor(w):
  def foo(features):
    return (features @ w > 0).astype(int)
  return foo


# Function that trains the model and returns prediction function
def train_model(y, x):
  w_init = np.zeros(x.shape[1])
  lambda_ = 1e-9
  weights, _ = reg_logistic_regression_sgd(
    y, x, lambda_, w_init, 400, 2000, 0.04,
  )
  return make_predictor(weights)


# Ensure same random number sequence
np.random.seed(21368342)

# Preprocess training data
tx_train_2, transformation_memory = transformation_pipeline(x_train)

# Training the model
predict = train_model(y_train, tx_train_2)

# Training metrics
train_acc = metrics.accuracy(y_train, predict(tx_train_2))
print(f'Training accuracy is {train_acc}')

# Training data removed to free RAM in order 
# to avoid issues encountered during experimenting
del tx_train_2
del y_train
gc.collect()
time.sleep(1)

# Loading test data
_, x_test, ids_test, cols_train = data_io.load_csv_data(f'{DATA_FILE_PREFIX}test.csv')

# Data is preprocessed and predicted in batches
# in order to make the best use of RAM
batch_size = 100000
current_index = 0
predictions_for_batches = []
while current_index <= x_test.shape[0]:
  x_test_batch = x_test[current_index:current_index + batch_size]
  current_index += batch_size
  tx_test_batch, _ = transformation_pipeline(x_test_batch, transformation_memory=transformation_memory)
  predictions = predict(tx_test_batch)
  predictions_for_batches.append(predictions)
  del tx_test_batch
  time.sleep(1)

# Collecting predictions for each batch into a single prediction array
collected_predictions = np.concatenate(predictions_for_batches)

# Map predictions from {0, 1} to {-1, 1}
collected_predictions = collected_predictions * 2 - 1

# Make sure you classified all test samples
assert(collected_predictions.shape[0] == x_test.shape[0])

# Creating submission
data_io.create_csv_submission(ids_test, collected_predictions, 'submission.csv')