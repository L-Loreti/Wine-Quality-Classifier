# import statsmodels as sm
import numpy as np

from sklearn.feature_selection import SequentialFeatureSelector
from statsmodels.tools.tools import add_constant

def get_best_features(model_declaration, model_name, xTrain, yTrain, MaxNumberOfFeatures, KFold_):

    best_features = []

    if model_name == 'LogRegression':

        xTrainLOGIT = add_constant(xTrain)

        for j in np.arange(1, MaxNumberOfFeatures + 1):
            sfs = SequentialFeatureSelector(model_declaration, n_features_to_select = j,
                                            direction = 'forward', scoring = 'accuracy', cv = KFold_)
            sfs.fit(xTrainLOGIT, yTrain)

            best_features.append(sfs.get_feature_names_out(xTrainLOGIT.columns.tolist()).tolist())

        best_features.append(xTrainLOGIT.columns.tolist())

    else:

        for j in np.arange(1, MaxNumberOfFeatures):

            sfs = SequentialFeatureSelector(model_declaration, n_features_to_select = j,
                                            direction = 'forward', scoring = 'accuracy', cv = KFold_)
            sfs.fit(xTrain, yTrain)

            best_features.append(sfs.get_feature_names_out(xTrain.columns.tolist()).tolist())
            
        best_features.append(xTrain.columns.tolist())

    return best_features


def read_features(file, model_names):

    lines = []
    best_features = []

    line = file.readline()
    lineList = line.split(', ')
    lineList = [item.strip() for item in lineList]

    # read and store all file lines as a list
    while line:

        lines.append(lineList)

        line = file.readline()
        lineList = line.split(', ')
        lineList = [item.strip() for item in lineList]

    # get the features for every model (algoithm not optimized)
    for i in range(len(model_names)):

        model = model_names[i]

        best_features_aux = []

        for j in range(len(lines)):

            if model == lines[j][0]:
                best_features_aux.append(lines[j][2::])

        best_features.append(best_features_aux)

    return best_features

def cost_function(fp1, fp2, p1, p2, q1, q2):
    return (q1*p1*fp1 + q2*(p2-p1)*fp2)