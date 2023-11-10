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


# print correlation matrix
sns.pairplot(df,hue='fracture')
plt.show()

sns.pairplot(df,hue='sex')
plt.show()
