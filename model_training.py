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

model_list = [LDA(), QDA(reg_param = 0.1), GaussianNB(), LogisticRegression(max_iter = 1000)]
model_names = ['LDA', 'QDA', 'GaussianNB', 'LogRegression']

file = open('[Best Features].txt', 'r')

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

n_folds = 8

kf = KFold(n_splits=n_folds, shuffle=True, random_state = 93)

xTrain = []
yTrain = []
xVal = []
yVal = []

for train_index, val_index in kf.split(x):

    xTrain.append(x.iloc[train_index]) 
    yTrain.append(y.iloc[train_index])

    xVal.append(x.iloc[val_index])
    yVal.append(y.iloc[val_index])


print('TREINAMENTO')
print('Classe 1:', len(xTrain[0].loc[yTrain[0] == 1]))
print('Classe 2:', len(xTrain[0].loc[yTrain[0] == 2]))
print('Classe 3:', len(xTrain[0].loc[yTrain[0] == 3]))
print(len(xVal[0].loc[yVal[0] == 3]))

###################
# MODELS TRAINING #
###################

n_features = len(xTrain[0].columns)
n_featuresLOGIT = n_features + 1

accuracy_class1 = np.zeros(shape=(len(model_list), len(xTrain[0].columns)+1, n_folds))
accuracy_class2 = np.zeros(shape=(len(model_list), len(xTrain[0].columns)+1, n_folds))
# accuracy_class3 = np.zeros(shape=(len(model_list), len(xTrain[0].columns)+1, n_folds))
accuracy = np.zeros(shape=(len(model_list), len(xTrain[0].columns)+1, n_folds))

for m in range(len(model_list)):

    model = model_list[m]

    if model_names[m] == 'LogRegression':

        for ft in range(n_featuresLOGIT):
            
            for fold in range(n_folds):

                xTrainLOGIT = add_constant(xTrain[fold])
                xValLOGIT = add_constant(xVal[fold])

                model.fit(xTrainLOGIT[best_features[m][ft]], yTrain[fold])
                predictions = model.predict(xValLOGIT[best_features[m][ft]])
                score = accuracy_score(predictions, yVal[fold])

                confMatrix = confusion_matrix(yVal[fold], predictions)

                accuracy[m][ft][fold] = score
                accuracy_class1[m][ft][fold] = confMatrix[0][0]/np.sum(confMatrix[0][:])
                accuracy_class2[m][ft][fold] = confMatrix[1][1]/np.sum(confMatrix[1][:])
                # accuracy_class3[m][ft][fold] = confMatrix[2][2]/np.sum(confMatrix[2][:])

    else:

        for ft in range(n_features):

            for fold in range(n_folds):

                model.fit(xTrain[fold][best_features[m][ft]], yTrain[fold])
                predictions = model.predict(xVal[fold][best_features[m][ft]])
                score = accuracy_score(predictions, yVal[fold])

                confMatrix = confusion_matrix(yVal[fold], predictions)

                accuracy[m][ft][fold] = score
                accuracy_class1[m][ft][fold] = confMatrix[0][0]/np.sum(confMatrix[0][:])
                accuracy_class2[m][ft][fold] = confMatrix[1][1]/np.sum(confMatrix[1][:])
                # accuracy_class3[m][ft][fold] = confMatrix[2][2]/np.sum(confMatrix[2][:])

######################################
# PLOT SCORE AVERAGES FOR EACH CLASS #
######################################

fig, axs = plt.subplots(2, 2, figsize = (16, 8))

mean_accuracy_class1 = np.zeros(shape = (len(model_list), n_featuresLOGIT))
std_accuracy_class1 = np.zeros(shape = (len(model_list), n_featuresLOGIT))

mean_accuracy_class2 = np.zeros(shape = (len(model_list), n_featuresLOGIT))
std_accuracy_class2 = np.zeros(shape = (len(model_list), n_featuresLOGIT))

# mean_accuracy_class3 = np.zeros(shape = (len(model_list), n_featuresLOGIT))
# std_accuracy_class3 = np.zeros(shape = (len(model_list), n_featuresLOGIT))

mean_accuracy_general = np.zeros(shape = (len(model_list), n_featuresLOGIT))
std_accuracy_general = np.zeros(shape = (len(model_list), n_featuresLOGIT))

for m in range(len(model_names)):

    for ft in range(n_featuresLOGIT):
        mean_accuracy_class1[m][ft] = np.mean(accuracy_class1[m][ft][:])
        std_accuracy_class1[m][ft] = np.std(accuracy_class1[m][ft][:])

        mean_accuracy_class2[m][ft] = np.mean(accuracy_class2[m][ft][:])
        std_accuracy_class2[m][ft] = np.std(accuracy_class2[m][ft][:])

        # mean_accuracy_class3[m][ft] = np.mean(accuracy_class3[m][ft][:])
        # std_accuracy_class3[m][ft] = np.std(accuracy_class3[m][ft][:])

        mean_accuracy_general[m][ft] = np.mean(accuracy[m][ft][:])
        std_accuracy_general[m][ft] = np.std(accuracy[m][ft][:])

for m in range(len(model_names)):
    axs[0, 0].errorbar(np.arange(1, n_featuresLOGIT + 1)+0.1*m, mean_accuracy_class1[m][:], std_accuracy_class1[m][:],
                 label = model_names[m], fmt = 'o')

    axs[0, 1].errorbar(np.arange(1, n_featuresLOGIT + 1)+0.1*m, mean_accuracy_class2[m][:], std_accuracy_class2[m][:],
                 label = model_names[m], fmt = 'o')
    
    # axs[1, 0].errorbar(np.arange(1, n_featuresLOGIT + 1)+0.1*m, mean_accuracy_class3[m][:], std_accuracy_class3[m][:],
    #              label = model_names[m], fmt = 'o')

    axs[1, 1].errorbar(np.arange(1, n_featuresLOGIT + 1)+0.1*m, mean_accuracy_general[m][:], std_accuracy_general[m][:],
                 label = model_names[m], fmt = 'o')

# fig.suptitle('test_size=' + str(test_size_) + 
#              ', folds=' + str(n_folds) +
#              ', size_label1=' + str(size_label1) + 
#              ', size_label2=' + str(size_label2) + 
#              ', size_label3=' + str(size_label3), fontsize=16)

axs[0, 0].set_title('Accuracy - Class 1')
axs[0, 0].set_yticks(np.arange(0,1.1,0.1))

axs[0, 1].set_title('Accuracy - Class 2')
axs[0, 1].set_yticks(np.arange(0,1.1,0.1))

# axs[1, 0].set_title('Accuracy - Class 3')
# axs[1, 0].set_yticks(np.arange(0,1.1,0.1))

axs[1, 1].set_title('Accuracy - General')
axs[1, 1].set_yticks(np.arange(0,1.1,0.1))

fig.tight_layout()
plt.legend()

# figname = ("test_size: " + str(test_size_) + ", Folds: " + str(n_folds) + 
#           ", size_label1: " + str(size_label1) +
#           ", size_label2: " + str(size_label2) +
#           ", size_label3: " + str(size_label3) + ".png")
# plt.savefig(figname, dpi = 600)

plt.show()