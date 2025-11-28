import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mysql import connector

from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

connection = connector.connect(
  host = '127.0.0.1',
  user = 'Leonardo-Loreti',
  password = '159753bhg',
  database = 'WineQT'
)

query = 'SELECT * FROM WineData'

wine = pd.read_sql(query, con = connection)
# Same data than the dataframe index
wine = wine.drop(['id'], axis = 1)

connection.close()

#############################
# EXPLORATORY DATA ANALYSIS #
#############################

# Check if there are null values or duplicated values
'''
print(wine.info())
print(wine.duplicated().sum())
'''
# Let's look at the quantity of data for each class
'''
bins_ = [i for i in np.arange(2.75,9.75)]

plt.hist(wine['quality'], bins = bins_,
         width = 0.5, edgecolor = 'k', color = '#2ca02c')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.xlim(2.5, 8.5)

figname = ("/home/leonardo/Documentos/Ciência de Dados/Wine-Quality-Classifier/figs-results/Unbalanced dataset.png")
plt.savefig(figname, dpi = 600)

plt.show()
'''
# There is a high inbalance between the classes quantity of data, so 
# let's join some classes

wine['quality'].loc[wine['quality'] == 3] = 1
wine['quality'].loc[wine['quality'] == 4] = 1
wine['quality'].loc[wine['quality'] == 5] = 1
wine['quality'].loc[wine['quality'] == 6] = 2
wine['quality'].loc[wine['quality'] == 7] = 2
wine['quality'].loc[wine['quality'] == 8] = 2
'''
# Plot histogram after classes change
plt.hist(wine['quality'].loc[wine['quality'] == 1], bins = [0.75, 1.75], edgecolor = 'k', color = '#2ca02c', width = 0.5)
plt.hist(wine['quality'].loc[wine['quality'] == 2], bins = [1.75, 2.75], edgecolor = 'k', color = '#2ca02c', width = 0.5)
plt.xlabel('Quality')
plt.ylabel('Count')
plt.xlim(0.5, 2.5)
plt.xticks([0.5, 1.0, 1.5, 2.0, 2.5])

figname = ("/home/leonardo/Documentos/Ciência de Dados/Wine-Quality-Classifier/figs-results/Balanced dataset.png")
plt.savefig(figname, dpi = 600)

plt.show()
'''
###############################################
# VIF ANALYSIS BEFORE FEATURES TRANSFORMATION #
###############################################
'''
# Adding constant for the VIF regression procedure
x = add_constant(wine)

VIF = pd.Series([variance_inflation_factor(x.values, i) for i in range(x.shape[1])], index=x.columns)
print(VIF)
'''
#############################################################
# PEARSON CORRELATION MATRIX BEFORE FEATURES TRANSFORMATION #
#############################################################
'''
# Pearson correlation coefficient matrix
corr = wine.corr()

# Generate a mask for the upper triangle
mask1 = np.triu(np.ones_like(corr, dtype=bool))

threshold = 0
mask2 = abs(corr) < threshold

# Draw the heatmap with the mask and correct aspect ratio

fig = plt.figure(figsize = (12,8))

sns.heatmap(corr, mask = mask1 | mask2, cmap='inferno', vmin = -1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.xticks(rotation=30, ha='right')

figname = ("/home/leonardo/Documentos/Ciência de Dados/Wine-Quality-Classifier/figs-results/heatmap_originalFeatures.png")
plt.savefig(figname, dpi = 600)

plt.show()
'''
############################
# VARIABLES TRANSFORMATION #
############################

wine['total acidity'] = wine['fixed acidity'] + wine['volatile acidity']
wine['citric acid percentage'] = wine['citric acid']/wine['total acidity']
wine['free sulfur dioxide percentage'] = wine['free sulfur dioxide']/wine['total sulfur dioxide']
wine['percentage of alcohol density'] = (wine['alcohol']/100)/wine['density']

quality_col = wine.pop('quality')
wine.insert(len(wine.columns), 'quality', quality_col)

wine = wine.drop(['fixed acidity', 'volatile acidity', 'citric acid',
                  'free sulfur dioxide', 'total sulfur dioxide',
                  'alcohol', 'density', 'pH'], axis = 1)

##############################################
# VIF ANALYSIS AFTER FEATURES TRANSFORMATION #
##############################################
'''
# Adding constant for the VIF regression procedure
x = add_constant(wine)

VIF = pd.Series([variance_inflation_factor(x.values, i) for i in range(x.shape[1])], index=x.columns)
print(VIF)
'''
############################################################
# PEARSON CORRELATION MATRIX AFTER FEATURES TRANSFORMATION #
############################################################

fig = plt.figure(figsize=(12,8))

# Pearson correlation coefficient matrix
corr = wine.corr()

# Generate a mask for the upper triangle
mask1 = np.triu(np.ones_like(corr, dtype=bool))

threshold = 0
mask2 = abs(corr) < threshold

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask = mask1 | mask2, cmap='inferno', vmin = -1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.xticks(rotation=30, ha = 'right')
plt.gcf().subplots_adjust(bottom = 0.168)

figname = ("/home/leonardo/Documentos/Ciência de Dados/Wine-Quality-Classifier/figs-results/heatmap_modifiedFeatures.png")
plt.savefig(figname, dpi = 600)

plt.show()

##################################
# SCATTER PLOTS FOR ALL FEATURES #
##################################
# It is always important to check if there are data patterns that can
# result in a low value for the Pearson Correlation coefficient, even
# if there is a strong relation between the variables. 
'''
axes = pd.plotting.scatter_matrix(wine.drop(['quality'], axis = 1), figsize = (16,8), color = '#2ca02c', edgecolors = 'k', linewidths = 0.5, 
                                  hist_kwds={'color':'#2ca02c', 'edgecolor':'k'})
for ax in axes.flatten():
  ax.xaxis.label.set_rotation(30)
  ax.yaxis.label.set_rotation(0)
  ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.gcf().subplots_adjust(wspace = 0, hspace = 0, left = 0.18, bottom = 0.201)

figname = ("/home/leonardo/Documentos/Ciência de Dados/Wine-Quality-Classifier/figs-results/scatter_plot_withoutTarget_modifiedFeatures.png")
plt.savefig(figname, dpi = 600)

plt.show()
'''
########################################################
# SCATTER PLOT FOR TARGET WITH THE OTHER FEATURES ONLY #
########################################################
'''
df_columns = wine.drop(['quality'], axis = 1).copy().columns.tolist()

n_rows = 2
n_cols = 4

fig, axs = plt.subplots(n_rows, n_cols, figsize = (16,8))

for i in range(n_rows):
  for j in range(n_cols):
    if i*n_cols + j != n_rows*n_cols - 1:
      axs[i, j].scatter(wine['quality'], wine[df_columns[i*n_cols + j]], s = 20, color = '#2ca02c', edgecolors = 'k', linewidths = 0.5)
      # axs[i, j].scatter(wine_normal['quality'], wine_normal[df_columns[i*n_cols + j]], s = 10)
      # axs[i, j].scatter(wine_outliers['quality'], wine_outliers[df_columns[i*n_cols + j]], 
      #                   color = 'red', s = 10)
      axs[i, j].set_xlabel('Quality')
      axs[i, j].set_ylabel(df_columns[i*n_cols + j])

plt.tight_layout()
plt.gcf().subplots_adjust(bottom = 0.07, hspace = 0.169)

figname = ("/home/leonardo/Documentos/Ciência de Dados/Wine-Quality-Classifier/figs-results/scatter_plot_withTarget_modifiedFeatures.png")
plt.savefig(figname, dpi = 600)

plt.show()
'''
###########################################
# CREATE CSV FILE WITH MODIFIED DATAFRAME #
###########################################

# wine.to_csv('wine_modified.csv', index=False)
