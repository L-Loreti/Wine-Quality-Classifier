import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as func

from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA,
     QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, accuracy_score)
from statsmodels.tools.tools import add_constant

#########################################
# LOAD THE BEST FEATURES FOR EACH MODEL #
#########################################

# models used for training
model_list = [LDA(), QDA(reg_param = 0.1), GaussianNB(), LogisticRegression(max_iter = 1000)]
model_names = ['LDA', 'QDA', 'GaussianNB', 'LogRegression']

file = open('###path_to/best_features.txt', 'r')

best_features = func.read_features(file, model_names)

file.close()

#####################################
# LOAD THE TRAINING/VALIDATION DATA #
#####################################

wine_train = pd.read_csv('wine_modified_train_validation.csv')

x = wine_train.drop(['quality'], axis = 1).copy()
y = wine_train['quality'].copy()

#############################################
# SPLIT TRAINING/VALIDATION DATA INTO FOLDS #
#############################################
# number of folds for cross-validation
n_folds = 10

kf = KFold(n_splits=n_folds, shuffle=True, random_state = 81)

xTrain = []
yTrain = []
xVal = []
yVal = []

# separate the training and validation data base on the KFold generator,
# with the same seed used to find the best features
for train_index, val_index in kf.split(x):

    xTrain.append(x.iloc[train_index]) 
    yTrain.append(y.iloc[train_index])

    xVal.append(x.iloc[val_index])
    yVal.append(y.iloc[val_index])

###################
# MODELS TRAINING #
###################

n_features = len(xTrain[0].columns)
n_featuresLOGIT = n_features + 1

# create the arrays to store the accucary for each class and the overall accuracy_score
# the first dimension of the array represents the different models; the second the number
# of features used for the training and validation (that's why there is a +1, and the 
# models which don't have a constant will have the last element of this dimension as null
# values); the last dimension represents the number of folds of cross-validation, allowing
# us to take an average of the accuracies and its standard deviation
accuracy_class1 = np.zeros(shape=(len(model_list), len(xTrain[0].columns)+1, n_folds))
accuracy_class2 = np.zeros(shape=(len(model_list), len(xTrain[0].columns)+1, n_folds))
accuracy = np.zeros(shape=(len(model_list), len(xTrain[0].columns)+1, n_folds))

for m in range(len(model_list)):

    # get a specific model
    model = model_list[m]

    # if the model is the Logistic Regression, run this loop, because we need to add a 
    # constant to the dataset
    if model_names[m] == 'LogRegression':

        # run through all the best features
        for ft in range(n_featuresLOGIT):
            
            for fold in range(n_folds):

                # add a constant to the training and validation dataset
                xTrainLOGIT = add_constant(xTrain[fold])
                xValLOGIT = add_constant(xVal[fold])

                # fit the model, get the predictions and the overall score
                model.fit(xTrainLOGIT[best_features[m][ft]], yTrain[fold])
                predictions = model.predict(xValLOGIT[best_features[m][ft]])
                score = accuracy_score(predictions, yVal[fold])

                # evaluate the confusion matrix, which will enable us to calculate
                # the score for each class
                confMatrix = confusion_matrix(yVal[fold], predictions)

                # store the overall score on its respective array
                accuracy[m][ft][fold] = score
                # calculate the accuracy of the classes as the number of TRUE POSITIVES
                # the respective class, divided by the total amount of data of the
                # respective class, represented by the lines of the confusion matrix
                accuracy_class1[m][ft][fold] = confMatrix[0][0]/np.sum(confMatrix[0, :])
                accuracy_class2[m][ft][fold] = confMatrix[1][1]/np.sum(confMatrix[1, :])

    else:

        # same thing as the previous loop, without adding a constant to the data
        # this works for the other models
        for ft in range(n_features):

            for fold in range(n_folds):

                model.fit(xTrain[fold][best_features[m][ft]], yTrain[fold])
                predictions = model.predict(xVal[fold][best_features[m][ft]])
                score = accuracy_score(predictions, yVal[fold])

                confMatrix = confusion_matrix(yVal[fold], predictions)

                accuracy[m][ft][fold] = score
                accuracy_class1[m][ft][fold] = confMatrix[0][0]/np.sum(confMatrix[0, :])
                accuracy_class2[m][ft][fold] = confMatrix[1][1]/np.sum(confMatrix[1, :])

accuracies_file = open('###path_to/Model_accuracies_different_classes_folds=' + str(n_folds) + '.txt', 'w')

# declare arrays to store the means and standard deviations for each class and
# the overall case.
# the first dimension represents the model, and the last dimension, the values
# of the mean and std for the training with different quantity of best features
mean_accuracy_class1 = np.zeros(shape = (len(model_list), n_featuresLOGIT))
std_accuracy_class1 = np.zeros(shape = (len(model_list), n_featuresLOGIT))

mean_accuracy_class2 = np.zeros(shape = (len(model_list), n_featuresLOGIT))
std_accuracy_class2 = np.zeros(shape = (len(model_list), n_featuresLOGIT))

mean_accuracy_general = np.zeros(shape = (len(model_list), n_featuresLOGIT))
std_accuracy_general = np.zeros(shape = (len(model_list), n_featuresLOGIT))

for m in range(len(model_names)):

    for ft in range(n_featuresLOGIT):
        # calculate the mean and standard deviations for each class, and the overall values
        mean_accuracy_class1[m][ft] = np.mean(accuracy_class1[m][ft][:])
        std_accuracy_class1[m][ft] = np.std(accuracy_class1[m][ft][:])

        mean_accuracy_class2[m][ft] = np.mean(accuracy_class2[m][ft][:])
        std_accuracy_class2[m][ft] = np.std(accuracy_class2[m][ft][:])

        mean_accuracy_general[m][ft] = np.mean(accuracy[m][ft][:])
        std_accuracy_general[m][ft] = np.std(accuracy[m][ft][:])

        # store these values on .txt's files.
        accuracies_file.write('[Class 1], ' + str(model_names[m]) + ', ' + str(ft+1) + ', ' 
                              + str(mean_accuracy_class1[m][ft]) + ', ' + str(std_accuracy_class1[m][ft]) + '\n')
        accuracies_file.write('[Class 2], ' + str(model_names[m]) + ', ' + str(ft+1) + ', ' 
                              + str(mean_accuracy_class2[m][ft]) + ', ' + str(std_accuracy_class2[m][ft]) + '\n')
        accuracies_file.write('[General], ' + str(model_names[m]) + ', ' + str(ft+1) + ', ' 
                              + str(mean_accuracy_general[m][ft]) + ', ' + str(std_accuracy_general[m][ft]) + '\n')

accuracies_file.close()

######################################
# PLOT SCORE AVERAGES FOR EACH CLASS #
######################################

# plot the accuracies for each class, and the overall accuracy
fig, axes = plt.subplot_mosaic([['left', 'right'],
                                ['bottom', 'bottom']],
                               layout='tight', figsize = (12,8))

# Plot on the axes using their names
axes['left'].set_title('Class 1')
axes['left'].set_ylabel('Accuracy')
axes['left'].set_ylim(-0.03, 1)
axes['left'].set_yticks(np.arange(0, 1.1, 0.1))
axes['left'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axes['right'].set_title('Class 2')
axes['right'].set_ylabel('Accuracy')
axes['right'].set_ylim(-0.03, 1)
axes['right'].set_yticks(np.arange(0, 1.1, 0.1))
axes['right'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axes['bottom'].set_title('General')
axes['bottom'].set_ylabel('Accuracy')
axes['bottom'].set_xlabel('Number of Features')
axes['bottom'].set_ylim(-0.03, 1)
axes['bottom'].set_yticks(np.arange(0, 1.1, 0.1))
axes['bottom'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

# shift the data points to visualize all of them
points_position = [-0.15, -0.05, 0.05, 0.15]

for m in range(len(model_names)):
    # plot the means and std for each class and the overall values
    axes['left'].errorbar(np.arange(1, n_featuresLOGIT + 1)+points_position[m], mean_accuracy_class1[m][:], std_accuracy_class1[m][:],
                 label = model_names[m], fmt = 'o')

    axes['right'].errorbar(np.arange(1, n_featuresLOGIT + 1)+points_position[m], mean_accuracy_class2[m][:], std_accuracy_class2[m][:],
                 label = model_names[m], fmt = 'o')

    axes['bottom'].errorbar(np.arange(1, n_featuresLOGIT + 1)+points_position[m], mean_accuracy_general[m][:], std_accuracy_general[m][:],
                 label = model_names[m], fmt = 'o')

fig.suptitle('Folds=' + str(n_folds),
             fontsize = 16)
plt.legend(loc = 'lower left')

# Save figure, if intended
figname = ("###path_to/Model Accuracies, folds=" + str(n_folds) + ".png")
plt.savefig(figname, dpi = 600)

plt.show()
