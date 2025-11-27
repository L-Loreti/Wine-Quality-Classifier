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

features = ['chlorides', 'sulphates', 'total acidity', 'free sulfur dioxide percentage', 'percentage of alcohol density']

lda = pickle.load(open('Trained model.pkl', 'rb'))

predictions = lda.predict(xTest[features])
pred_proba = lda.predict_proba(xTest[features])
pred_proba_class1 = pred_proba[:,0]

print(lda.coef_)
print(lda.get_params(deep = True))

'''
confMatrix = confusion_matrix(yTest, predictions)

fpr, tpr, thresholds = roc_curve(y_true = yTest, y_score = pred_proba_class1,  pos_label = 1)

fig, axs = plt.subplot_mosaic([['top-left', 'top-right'],
                               ['middle-left', 'middle-right'],
                               ['bottom', 'bottom']],
                              layout='tight', figsize = (16,8))

axs['top-left'].set_ylabel('Cost Function (R$)')
axs['top-left'].set_xlabel('Class 1 Threshold')
axs['top-left'].set_xticks(np.arange(0, 1.1, 0.1))
axs['top-left'].set_xlim(0, 1)
axs['top-left'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axs['top-right'].set_ylabel('% of False Positives - Class 2')
axs['top-right'].set_xlabel('% of False Positives - Class 1')
axs['top-right'].set_xticks(np.arange(0, 1.1, 0.1))
axs['top-right'].set_yticks(np.arange(0, 1.1, 0.1))
axs['top-right'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)
axs['top-right'].set_ylim(0, 0.5)

axs['middle-left'].set_ylabel('Accuracy - Class 1')
axs['middle-left'].set_xlabel('Class 1 Threshold')
axs['middle-left'].set_xticks(np.arange(0, 1.1, 0.1))
axs['middle-left'].set_yticks(np.arange(0, 1.1, 0.1))
axs['middle-left'].set_xlim(0, 1)
axs['middle-left'].set_ylim(0.4, 1)
axs['middle-left'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axs['middle-right'].set_ylabel('Accuracy - Class 2')
axs['middle-right'].set_xlabel('Class 1 Threshold')
axs['middle-right'].set_xticks(np.arange(0, 1.1, 0.1))
axs['middle-right'].set_yticks(np.arange(0, 1.1, 0.1))
axs['middle-right'].set_xlim(0, 1)
axs['middle-right'].set_ylim(0.5, 1)
axs['middle-right'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

axs['bottom'].set_ylabel('General Accuracy')
axs['bottom'].set_xlabel('Class 1 Threshold')
axs['bottom'].set_xticks(np.arange(0, 1.1, 0.1))
axs['bottom'].set_xlim(0, 1)
axs['bottom'].set_ylim(0.4, 1)
axs['bottom'].grid(visible = True, axis = 'y', linestyle = '--', linewidth=0.7)

p1 = 45
p2 = 90
q1 = 3000
q2 = 1000

maxScore = 0
scoreClass1_maxScoreThreshold = 0
scoreClass2_maxScoreThreshold = 0
maxScoreThreshold = 0
refund = 0
amount_sales = p1*q1 + p2*q2

for t in thresholds:
    pred_threshold = []

    for i in pred_proba_class1:
        if i > t:
            pred_threshold.append(1)
        else:
            pred_threshold.append(2)

    confMatrix_threshold = confusion_matrix(yTest, pred_threshold)

    fp1 = confMatrix_threshold[1][0]/np.sum(confMatrix_threshold[:, 0])
    fp2 = confMatrix_threshold[0][1]/np.sum(confMatrix_threshold[:, 1])

    accuracy_class1 = confMatrix_threshold[0][0]/np.sum(confMatrix_threshold[:, 0])
    accuracy_class2 = confMatrix_threshold[1][1]/np.sum(confMatrix_threshold[:, 1])
    score = accuracy_score(pred_threshold, yTest)
    cost = func.cost_function(fp1, fp2, p1, p2, q1, q2)

    if score > maxScore:
        maxScore = score
        scoreClass1_maxScoreThreshold = accuracy_class1
        scoreClass2_maxScoreThreshold = accuracy_class2
        maxScoreThreshold = t
        refund = cost

    axs['top-left'].scatter(t, cost, color = 'k', s = 12)
    axs['top-right'].scatter(fp1, fp2, color = 'k', s = 12)
    axs['middle-left'].scatter(t, accuracy_class1, color = 'k', s = 12)
    axs['middle-right'].scatter(t, accuracy_class2, color = 'k', s = 12)
    axs['bottom'].scatter(t, score, color = 'k', s = 12)

fig.suptitle('Fine-tuning', fontsize = 16)

plt.show()

print("Maximum General Accuracy:", np.round(maxScore, 3), ", occurs when class 1 threshold is:", np.round(maxScoreThreshold, 3))
print("Class 1 accuracy:", np.round(scoreClass1_maxScoreThreshold, 3))
print("Class 2 accuracy:", np.round(scoreClass2_maxScoreThreshold, 3))
print("Total ammount of sales is: R$", np.round(amount_sales, 2))
print("Refund is: R$", np.round(refund, 2), "which is", np.round(refund/amount_sales, 3)*100, "% of sales" )
'''