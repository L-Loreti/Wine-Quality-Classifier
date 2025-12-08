import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import functions as func

from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, 
                             RocCurveDisplay, roc_curve, accuracy_score)

####################################
# LOAD THE TEST DATA AND THE MODEL #
####################################

wine_test = pd.read_csv('wine_modified_test.csv')

xTest = wine_test.drop(['quality'], axis = 1).copy()
yTest = wine_test['quality'].copy()

# best features of the best model
features = ['residual sugar', 'chlorides', 'sulphates', 'total acidity', 'free sulfur dioxide percentage', 'percentage of alcohol density']

# load the best model already trained
model = pickle.load(open('Trained model.pkl', 'rb'))

# obtain the predictions and explicit probabilities to get the values for class 1
predictions = model.predict(xTest[features])
pred_proba = model.predict_proba(xTest[features])
pred_proba_class1 = pred_proba[:,0]

# evaluate the confusion matrix
confMatrix_nonOptm = confusion_matrix(yTest, predictions)

# calculate the false positives for each class as the number of false positives
# for each class: second diagonal elements (false positives)/the total amount
# of elements for each class
fp1_nonOptm = confMatrix_nonOptm[1][0]/np.sum(confMatrix_nonOptm[1, :])
fp2_nonOptm = confMatrix_nonOptm[0][1]/np.sum(confMatrix_nonOptm[0, :])

# calculate the accuracies for each class, as done on the training and testing procedure
accuracy_class1_nonOptm = confMatrix_nonOptm[0][0]/np.sum(confMatrix_nonOptm[0, :])
accuracy_class2_nonOptm = confMatrix_nonOptm[1][1]/np.sum(confMatrix_nonOptm[1, :])

#################
# BUSINESS CASE #
#################

    #######################
    # NON-OPTIMIZED MODEL #
    #######################

# average price of class 1 wines
p1 = 45
# average price of class 2 wines
p2 = 70
# quantity sold of class 1 wines
q1 = 1000
# quantity sold of class 2 wines
q2 = 1500
# total amount of sales
amount_sales = p1*q1 + p2*q2

# score and cost of the non-optimized algorithm
score_nonOptmz = accuracy_score(predictions, yTest)
cost_nonOptm = func.cost_function(fp1_nonOptm, fp2_nonOptm, p1, p2, q1, q2)

# Metrics for the non-optimized algorithm. Just uncomment some of them, if needed
print("[NON-OPTIMIZED ALGORITHM]")
print("General Accuracy:", np.round(score_nonOptmz, 3))
print("Class 1 accuracy:", np.round(accuracy_class1_nonOptm, 3))
print("Class 2 accuracy:", np.round(accuracy_class2_nonOptm, 3))
# FPR are the False Positive Rates
print("FPR 1:", np.round(fp1_nonOptm, 3))
print("FPR 2:", np.round(fp2_nonOptm, 3))
# print("Balanced Accuracy - Class 1:", np.round((accuracy_class1_nonOptm + fp1_nonOptm)/2, 3))
# print("Balanced Accuracy - Class 2:", np.round((accuracy_class2_nonOptm + fp2_nonOptm)/2, 3))
print("Total ammount of sales is: R$", np.round(amount_sales, 2))
print("Refund is: R$", np.round(cost_nonOptm, 2), "which is", np.round(cost_nonOptm*100/amount_sales, 2), "% of sales" )

# Display ROC-AUC curve
RocCurveDisplay.from_predictions(yTest, pred_proba_class1, pos_label = 1, color = '#2ca02c')
plt.plot([0, 1], [0, 1], '--', alpha = 0.7, linewidth = 1, color = 'k')
plt.ylim(0,1)
plt.xlim(0,1)

# Save figure, if needed
figname = ("###path_to/AUC_ROC_curve.png")
plt.savefig(figname, dpi = 600)

plt.show()

    #############################
    # OPTIMIZATION OF THE MODEL #
    #############################

# get the false positive and true positive rates, and the boundary thresholds
fpr, tpr, thresholds = roc_curve(y_true = yTest, y_score = pred_proba_class1,  pos_label = 1)

# declare the figure and axes elements to plot
fig, axs = plt.subplot_mosaic([['top-left', 'top-right'],
                               ['secondLine-left', 'secondLine-right'],
                               ['bottom', 'bottom']],
                              layout='tight', figsize = (12,8))

axs['top-left'].set_ylabel('Cost Function (R$)')
axs['top-left'].set_xlabel('Class 1 Threshold')
axs['top-left'].set_xticks(np.arange(0, 1.1, 0.1))
axs['top-left'].set_xlim(0, 1)
axs['top-left'].set_ylim(25000, 100000)
axs['top-left'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axs['top-right'].set_ylabel('FPR - Class 2')
axs['top-right'].set_xlabel('FPR - Class 1')
axs['top-right'].set_xticks(np.arange(0, 1.1, 0.1))
axs['top-right'].set_yticks(np.arange(0, 1.1, 0.1))
axs['top-right'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)
axs['top-right'].set_ylim(0, 1)

axs['secondLine-left'].set_ylabel('FPR - Class 1')
axs['secondLine-left'].set_xlabel('Class 1 Threshold')
axs['secondLine-left'].set_xticks(np.arange(0, 1.1, 0.1))
axs['secondLine-left'].set_yticks(np.arange(0, 1.1, 0.1))
axs['secondLine-left'].set_xlim(0, 1)
axs['secondLine-left'].set_ylim(0, 1)
axs['secondLine-left'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axs['secondLine-right'].set_ylabel('FPR - Class 2')
axs['secondLine-right'].set_xlabel('Class 1 Threshold')
axs['secondLine-right'].set_xticks(np.arange(0, 1.1, 0.1))
axs['secondLine-right'].set_yticks(np.arange(0, 1.1, 0.1))
axs['secondLine-right'].set_xlim(0, 1)
axs['secondLine-right'].set_ylim(0, 1)
axs['secondLine-right'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axs['bottom'].set_ylabel('General Accuracy')
axs['bottom'].set_xlabel('Class 1 Threshold')
axs['bottom'].set_xticks(np.arange(0, 1.1, 0.1))
axs['bottom'].set_xlim(0, 1)
axs['bottom'].set_ylim(0.4, 1)
axs['bottom'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

        ################################################
        # SEARCH FOR THE BEST CLASSIFICATION THRESHOLD #
        ################################################

# initialize the variables that will store the values of the maximum
# overall and classes score, the false positive values, the threshold
# itself and the total amount of sales on the optimum threshold
maxScore = 0
scoreClass1_maxScoreThreshold = 0
scoreClass2_maxScoreThreshold = 0
fprClass1_max_ScoreThreshold = 0
fprClass2_max_ScoreThreshold = 0
maxScoreThreshold = 0
amount_sales = p1*q1 + p2*q2
# Initialize the refund with a value greater than the amount of sales
refund = (p1*q1 + p2*q2)*10

# invert the threshold values to start from the smaller one
thresholds_inverse = thresholds[::-1]

# search the optimum threshold using all the values given by the ROC curve
for t in thresholds_inverse:
    pred_threshold = []

    # apply different thresholds and get its predictions
    for i in pred_proba_class1:
        if i > t:
            pred_threshold.append(1)
        else:
            pred_threshold.append(2)

    # evaluate the confusion matrix to calculate the false positives
    confMatrix_threshold = confusion_matrix(yTest, pred_threshold)

    # calculate the false positives as done previously
    fp1 = confMatrix_threshold[1][0]/np.sum(confMatrix_threshold[1, :])
    fp2 = confMatrix_threshold[0][1]/np.sum(confMatrix_threshold[0, :])

    # calcualte the accuracies for each class, the overall accuracy and the cost
    accuracy_class1 = confMatrix_threshold[0][0]/np.sum(confMatrix_threshold[0, :])
    accuracy_class2 = confMatrix_threshold[1][1]/np.sum(confMatrix_threshold[1, :])
    
    score = accuracy_score(pred_threshold, yTest)
    cost = func.cost_function(fp1, fp2, p1, p2, q1, q2)

    # condition to find the threshold based on the minimum cost
    if cost < refund:
        maxScore = score
        scoreClass1_maxScoreThreshold = accuracy_class1
        scoreClass2_maxScoreThreshold = accuracy_class2
        fprClass1_max_ScoreThreshold = fp1
        fprClass2_max_ScoreThreshold = fp2
        maxScoreThreshold = t
        refund = cost

    # plot the cost values, the false positives for each classe and the overall accuracy, for different
    # values of the threshold
    axs['top-left'].scatter(t, cost, color = '#2ca02c', edgecolor = 'k', s = 22, linewidths = 0.5)
    axs['top-right'].scatter(fp1, fp2, color = '#2ca02c', edgecolor = 'k', s = 22, linewidths = 0.5)
    axs['secondLine-left'].scatter(t, fp1, color = '#2ca02c', edgecolor = 'k', s = 22, linewidths = 0.5)
    axs['secondLine-right'].scatter(t, fp2, color = '#2ca02c', edgecolor = 'k', s = 22, linewidths = 0.5)
    axs['bottom'].scatter(t, score, color = '#2ca02c', edgecolor = 'k', s = 22, linewidths = 0.5)

fig.suptitle('Fine-tuning', fontsize = 16)

# Save figure, if needed
figname = ("###path_to/fine_tuning.png")
plt.savefig(figname, dpi = 600)

plt.show()

# Metrics of the optimized algorithm. Just uncomment some of them if needed
print("\n[OPTIMIZED ALGORITHM]")
print("Maximum General Accuracy:", np.round(maxScore, 4), ", occurs when class 1 threshold is:", np.round(maxScoreThreshold, 3))
# print("Class 1 accuracy:", np.round(scoreClass1_maxScoreThreshold, 3))
# print("Class 2 accuracy:", np.round(scoreClass2_maxScoreThreshold, 3))
print("FPR 1:", np.round(fprClass1_max_ScoreThreshold, 3))
print("FPR 2:", np.round(fprClass2_max_ScoreThreshold, 3))
# print("Balanced Accuracy - Class 1:", np.round((scoreClass1_maxScoreThreshold + fprClass1_max_ScoreThreshold)/2, 3))
# print("Balanced Accuracy - Class 2:", np.round((scoreClass2_maxScoreThreshold + fprClass2_max_ScoreThreshold)/2, 3))
print("Total ammount of sales is: R$", np.round(amount_sales, 2))
print("Refund is: R$", np.round(refund, 2), "which is", np.round(refund*100/amount_sales, 2), "% of sales" )