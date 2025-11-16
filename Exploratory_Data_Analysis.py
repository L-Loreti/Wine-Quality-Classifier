import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

wine = pd.read_csv('/content/gdrive/MyDrive/Ciência de Dados/Qualidade do Vinho - Classificação/WineQT.csv')
# Same data than the dataframe index
wine = wine.drop(['Id'], axis = 1)

#############################
# EXPLORATORY DATA ANALYSIS #
#############################

# Check if there are null values or duplicated values
'''
print(wine.info())
print(wine.duplicated().sum())
'''

# It is always important to check if there are data patterns that can
# result in a low value for the Pearson Correlation coefficient, even
# if there is a strong relation between the variables. 
'''
pd.plotting.scatter_matrix(wine, figsize = (22,22))
'''

# It is possible to realize some outliers by looking at the features
# scatterplots with the 'quality' index

wine_outliers = wine.loc[wine['total sulfur dioxide'] > 250].copy()
wine_normal = wine.loc[wine['total sulfur dioxide'] < 250].copy()

df_columns = wine.drop(['quality'], axis = 1).copy().columns.tolist()

n_rows = 4
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, figsize = (12, 12))

for i in range(n_rows):
  for j in range(n_cols):
    if i*n_cols + j != n_rows*n_cols - 1:
      axs[i, j].scatter(wine_normal['quality'], wine_normal[df_columns[i*n_cols + j]], s = 10)
      axs[i, j].scatter(wine_outliers['quality'], wine_outliers[df_columns[i*n_cols + j]], 
                        color = 'red', s = 10)
      axs[i, j].set_xlabel('Quality')
      axs[i, j].set_ylabel(df_columns[i*n_cols + j])

plt.tight_layout()
plt.show()


# Pearson correlation coefficient matrix
corr = wine.corr()

# Generate a mask for the upper triangle
mask1 = np.triu(np.ones_like(corr, dtype=bool))

threshold = 0.4
mask2 = abs(corr) < threshold

# Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask = mask1 | mask2, cmap='inferno', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Adding constant for the VIF regression procedure
'''
x = add_constant(wine.drop(['Id'], axis = 1))

VIF = pd.Series([variance_inflation_factor(x.values, i) for i in range(x.shape[1])], index=x.columns)
print(VIF)
'''

#############################
# EXPLORATORY DATA ANALYSIS #
#############################
'''
wine = pd.read_csv('/content/gdrive/MyDrive/Ciência de Dados/Qualidade do Vinho - Classificação/WineQT.csv')

wine_correct = wine.loc[wine['total sulfur dioxide'] < 250]

wine_correct['total acidity'] = wine_correct['fixed acidity'] + wine_correct['volatile acidity']
wine_correct['citric acid percentage'] = wine_correct['citric acid']/wine_correct['total acidity']
wine_correct['free sulfur dioxide percentage'] = wine_correct['free sulfur dioxide']/wine_correct['total sulfur dioxide']
wine_correct['percentage of alcohol density'] = (wine_correct['alcohol']/100)/wine_correct['density']

wine_correct = wine_correct.drop(['Id','fixed acidity', 'volatile acidity', 'citric acid',
                                'free sulfur dioxide', 'total sulfur dioxide',
                                'alcohol', 'density', 'pH'], axis = 1)

# Calculation of VIF
x = wine_correct.drop(['quality'], axis = 1)
y = wine_correct['quality']

# Adding constant for the VIF regression procedure
x = add_constant(x)

VIF = pd.Series([variance_inflation_factor(x.values, i) for i in range(x.shape[1])], index=x.columns)
print(VIF)

# x = wine_correct.drop(['quality'], axis = 1)
# y = wine_correct['quality']

# y.loc[y == 3] = 1
# y.loc[y == 4] = 1
# y.loc[y == 5] = 2
# y.loc[y == 6] = 2
# y.loc[y == 7] = 3
# y.loc[y == 8] = 3
'''
# Plot histogram of quantity of data per quality
'''
plt.hist(y)
plt.show()
'''