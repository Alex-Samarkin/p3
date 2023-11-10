# base modules
import numpy as np
import scipy as sp
import pandas as pd
import statistics as st
# graphics
import matplotlib.pyplot as plt
import seaborn as sns

# read data
file = "./Logit_bone_densitometries.csv"
df = pd.read_csv(file,decimal='.',encoding='utf-8',sep=',')

# print data
print("Shape of dataset: ",df.shape)
print("Columns is: ",df.columns)

# print first and last 10 rows
print(df.head(n=10))
print(df.tail(n=10))

# print statistics
print(df.describe())

df['sex']=df['sex'].astype('category')
df['fracture'] = df['fracture'].astype('category')

# linear regression
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Get the independent variable X and dependent variable y from the dataframe
X = df['weight_kg'].values.reshape(-1, 1)
y = df['height_cm'].values

# Add a constant term to the independent variable matrix
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())


# Plot the regression line using seaborn
sns.regplot(x=X[:, 1], y=y, color='blue', label='Regression Line')
plt.show()