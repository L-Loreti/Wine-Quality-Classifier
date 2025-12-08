import numpy as np

from sklearn.feature_selection import SequentialFeatureSelector
from statsmodels.tools.tools import add_constant

# function that finds the best features for different models, searching in a forward manner
def get_best_features(model_declaration, model_name, xTrain, yTrain, MaxNumberOfFeatures, KFold_):

    best_features = []

    # we need a different loop for the Logistic Regression algorithm because it needs a constant
    # on the dataset
    if model_name == 'LogRegression':

        # add the constant
        xTrainLOGIT = add_constant(xTrain)

        # search the best features with the SequentialFeatureSelector and store it on the best_features
        # list
        for j in np.arange(1, MaxNumberOfFeatures + 1):
            sfs = SequentialFeatureSelector(model_declaration, n_features_to_select = j,
                                            direction = 'forward', scoring = 'accuracy', cv = KFold_)
            sfs.fit(xTrainLOGIT, yTrain)

            best_features.append(sfs.get_feature_names_out(xTrainLOGIT.columns.tolist()).tolist())

        best_features.append(xTrainLOGIT.columns.tolist())

    else:

        # the model isn't the Logistic Regression, this loop is executed
        for j in np.arange(1, MaxNumberOfFeatures):

            sfs = SequentialFeatureSelector(model_declaration, n_features_to_select = j,
                                            direction = 'forward', scoring = 'accuracy', cv = KFold_)
            sfs.fit(xTrain, yTrain)

            best_features.append(sfs.get_feature_names_out(xTrain.columns.tolist()).tolist())
            
        best_features.append(xTrain.columns.tolist())

    return best_features

# function that reads the best features from the best_features.txt file
# and returns a list with these names
def read_features(file, model_names):

    lines = []
    best_features = []

    # read line and split all the elements using the delimiter ', '
    # we need to do the first one manually, and then run a "while" loop
    line = file.readline()
    lineList = line.split(', ')
    lineList = [item.strip() for item in lineList]

    # read and store all file lines as a list
    while line:

        lines.append(lineList)

        line = file.readline()
        lineList = line.split(', ')
        lineList = [item.strip() for item in lineList]

    # get the features for every model
    for i in range(len(model_names)):

        model = model_names[i]

        best_features_aux = []

        for j in range(len(lines)):

            if model == lines[j][0]:
                # read from the second element on, which is the beggining of the feature names
                best_features_aux.append(lines[j][2::])

        best_features.append(best_features_aux)

    return best_features

# a simple cost function
def cost_function(fp1, fp2, p1, p2, q1, q2):
    return (q1*p1*fp1 + q2*(p2)*fp2)