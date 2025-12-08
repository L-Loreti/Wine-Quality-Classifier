import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

import functions as func

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score)

################################
# LOAD MODEL AND BEST FEATURES #
################################

# the best features for the best model
features = ['residual sugar', 'chlorides', 'sulphates', 'total acidity', 'free sulfur dioxide percentage', 'percentage of alcohol density']
# the best model
model = LDA()

#####################################
# LOAD THE TRAINING/VALIDATION DATA #
#####################################

wine_train = pd.read_csv('wine_modified_train_validation.csv')
wine_test = pd.read_csv('wine_modified_test.csv')

xTrain_Val = wine_train.drop(['quality'], axis = 1).copy()
yTrain_Val = wine_train['quality'].copy()

xTest = wine_test.drop(['quality'], axis = 1).copy()
yTest = wine_test['quality'].copy()

########################
# TRAIN AND TEST MODEL #
########################

# The LDA model doesn't support partial fit for us to split the data
# and train the model. So, we need to train it on the whole training/
# validation dataset
model.fit(xTrain_Val[features], yTrain_Val)

# get predictions, an the overall score
predictions = model.predict(xTest[features])
score = accuracy_score(predictions, yTest)

# evaluate the confusion matrix
confMatrix = confusion_matrix(yTest, predictions)

# calculate the accuracies for each class with the same manner we did
# on the training procedure
accuracy_class1 = confMatrix[0][0]/np.sum(confMatrix[0, :])
accuracy_class2 = confMatrix[1][1]/np.sum(confMatrix[1, :])

# print accuracies to check if the values are consistent with the previous
# training, because here we are not using the KFold splitting
print('General accuracy:', score)
print('Class 1 accuracy:', accuracy_class1)
print('Class 2 accuracy:', accuracy_class2)

# Create a file to store the trained model with permission to 
# write binary code
file_model = open('Trained model.pkl', 'wb')
pickle.dump(model, file_model)
file_model.close()
