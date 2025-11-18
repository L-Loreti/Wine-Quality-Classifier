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