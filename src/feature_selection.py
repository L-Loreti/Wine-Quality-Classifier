import pandas as pd
import functions as func
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA,
     QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

###################################################
# DB IMPORT AND SPLIT INTO TRAINING AND TEST SETS #
###################################################

wine = pd.read_csv('wine_modified.csv')

x = wine.drop(['quality'], axis = 1).copy()
y = wine['quality']

# split 25% of data to testing procedure
test_size_ = 0.25
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = test_size_, random_state = 42)

# make a copy of the training/validation dataset to prepare it for training
wine_modified_train_validation = xTrain.copy()
wine_modified_test = xTest.copy()

wine_modified_train_validation['quality'] = yTrain
wine_modified_test['quality'] = yTest

# Save the training and test dataframes
wine_modified_train_validation.to_csv('wine_modified_train_validation.csv', index = False)
wine_modified_test.to_csv('wine_modified_test.csv', index = False)

# Models list
model_list = [LDA(), QDA(reg_param = 0.1), GaussianNB(), LogisticRegression(max_iter = 1000)]
model_names = ['LDA', 'QDA', 'GaussianNB', 'LogRegression']

####################################################
# GET BEST FEATURES WITH FORWARD FEATURE SELECTION #
####################################################
# n of folds for cross validation
n_folds = 10

# n_splits_-fold cross-validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state = 81)

# the training of the logistic regression algorithm needs a constant,
# that's the reason for the +1
n_features = len(xTrain.columns)
n_featuresLOGIT = n_features + 1

# list where the best features will be stored
best_features = []

# find best features based on the function get_best_features, developed on the function.py
# file. Basically, it uses the SequentialFeatureSelector function from the sklearn library
for i in range(len(model_list)):
    best_features.append(func.get_best_features(model_list[i], model_names[i], xTrain, yTrain, n_features, kf))

####################################
# WRITE BEST FEATURES ON .TXT FILE #
####################################

file_features = open('###path_to/best_features.txt', 'w')

# store the best features on a .txt file 
for m in range(len(model_list)):
    if model_names[m] == 'LogRegression':
        for i in range(n_featuresLOGIT):
            string = model_names[m] + ', ' + str(i+1)

            for j in range(len(best_features[m][i])):
                string += ', ' + str(best_features[m][i][j])
            
            file_features.write(string + '\n')
    else:
        for i in range(n_features):
            string = model_names[m] + ', ' + str(i+1)

            for j in range(len(best_features[m][i])):
                string += ', ' + str(best_features[m][i][j])
            
            file_features.write(string + '\n')

file_features.close()