import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

import functions as func

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score)

################################
# LOAD MODEL AND BEST FEATURES #
################################

features = ['chlorides', 'sulphates', 'total acidity', 'percentage of alcohol density']
model = GaussianNB()

#####################################
# LOAD THE TRAINING/VALIDATION DATA #
#####################################

wine_train = pd.read_csv('wine_modified_train_validation.csv')
wine_test = pd.read_csv('wine_modified_test.csv')

xTrain_Val = wine_train.drop(['quality'], axis = 1).copy()
yTrain_Val = wine_train['quality'].copy()

xTest = wine_test.drop(['quality'], axis = 1).copy()
yTest = wine_test['quality'].copy()

#############################################
# SPLIT TRAINING/VALIDATION DATA INTO FOLDS #
#############################################

n_folds = 10

kf = KFold(n_splits=n_folds, shuffle=True, random_state = 81)

xTrain = []
yTrain = []

for train_index, val_index in kf.split(xTrain_Val):

    xTrain.append(xTrain_Val.iloc[train_index]) 
    yTrain.append(yTrain_Val.iloc[train_index])

########################
# TRAIN AND TEST MODEL #
########################

for f in range(n_folds):
    model.fit(xTrain[f][features], yTrain[f])

predictions = model.predict(xTest[features])
pred_proba = model.predict_proba(xTest[features])
score = accuracy_score(predictions, yTest)

confMatrix = confusion_matrix(yTest, predictions)

accuracy_class1 = confMatrix[0][0]/np.sum(confMatrix[:,0])
accuracy_class2 = confMatrix[1][1]/np.sum(confMatrix[:,1])

print('General accuracy:', score)
print('Class 1 accuracy:', accuracy_class1)
print('Class 2 accuracy:', accuracy_class2)

# Create a file to store the trained model with permission to 
# write binary code
file_model = open('Trained model.pkl', 'wb')
pickle.dump(model, file_model)
file_model.close()
