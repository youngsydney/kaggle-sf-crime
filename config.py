"""
Holds constants and file paths for the whole program
"""

import os

# Root path for the project
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# filepath to the training data
TRAIN_FP = ROOT_PATH + '/sfcrime_train.csv'

# filepath to the testing data
TEST_FP = ROOT_PATH + '/sfcrime_test.csv'

# filepath for the output results
OUT_FP = ROOT_PATH + '/syoung_output.csv'

# number of folds (K) for the K-fold validation
K = 3

# Type of Classifier you would like
# DecisionTree, RF, AB, Log, SVM
classifier = 'DecisionTree'

# the features to be used in the classifiers
features = []
# the categorical features that need to be encoded
categorical = ['PdDistrict', 'DayOfWeek']

#initially there are 2063 unique street names
# min = 2000 138 street names
str_min = 2000
