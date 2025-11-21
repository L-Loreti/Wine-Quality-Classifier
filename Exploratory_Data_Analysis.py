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
  password = '########',
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
plt.hist(wine['quality'])
plt.xlabel('Quality')
plt.ylabel('Quantity')
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

# Plot histogram after classes change

plt.hist(wine['quality'])
plt.xlabel('Quality')
plt.ylabel('Quantity')
plt.show()

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

threshold = 0.3
mask2 = abs(corr) < threshold

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask = mask1 | mask2, cmap='inferno', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
'''
############################
# VARIABLES TRANSFORMATION #
############################
'''
wine['total acidity'] = wine['fixed acidity'] + wine['volatile acidity']
wine['citric acid percentage'] = wine['citric acid']/wine['total acidity']
wine['free sulfur dioxide percentage'] = wine['free sulfur dioxide']/wine['total sulfur dioxide']
wine['percentage of alcohol density'] = (wine['alcohol']/100)/wine['density']

quality_col = wine.pop('quality')
wine.insert(len(wine.columns), 'quality', quality_col)

wine = wine.drop(['fixed acidity', 'volatile acidity', 'citric acid',
                  'free sulfur dioxide', 'total sulfur dioxide',
                  'alcohol', 'density', 'pH'], axis = 1)
'''
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
'''
# Pearson correlation coefficient matrix
corr = wine.corr()

# Generate a mask for the upper triangle
mask1 = np.triu(np.ones_like(corr, dtype=bool))

threshold = 0.4
mask2 = abs(corr) < threshold

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask = mask1 | mask2, cmap='inferno', vmin = -1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
'''
#############################################
# SCATTER PLOTS FOR ALL FEATURES AND TARGET #
#############################################
# It is always important to check if there are data patterns that can
# result in a low value for the Pearson Correlation coefficient, even
# if there is a strong relation between the variables. 
'''
pd.plotting.scatter_matrix(wine, figsize = (22,22))
plt.show()
'''
########################################################
# SCATTER PLOT FOR TARGET WITH THE OTHER FEATURES ONLY #
########################################################
'''
df_columns = wine.drop(['quality'], axis = 1).copy().columns.tolist()

n_rows = 2
n_cols = 4

fig, axs = plt.subplots(n_rows, n_cols, figsize = (12, 12))

for i in range(n_rows):
  for j in range(n_cols):
    if i*n_cols + j != n_rows*n_cols - 1:
      axs[i, j].scatter(wine['quality'], wine[df_columns[i*n_cols + j]], s = 10)
      # axs[i, j].scatter(wine_normal['quality'], wine_normal[df_columns[i*n_cols + j]], s = 10)
      # axs[i, j].scatter(wine_outliers['quality'], wine_outliers[df_columns[i*n_cols + j]], 
      #                   color = 'red', s = 10)
      axs[i, j].set_xlabel('Quality')
      axs[i, j].set_ylabel(df_columns[i*n_cols + j])

plt.tight_layout()
plt.show()
'''
###########################################
# CREATE CSV FILE WITH MODIFIED DATAFRAME #
###########################################

# wine.to_csv('wine_modified.csv', index=False)
