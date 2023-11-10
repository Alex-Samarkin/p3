# Python
# import modules
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import os

# read local or remote file
def read_file(local:str,url_file:str) -> pd.DataFrame:
    df = pd.DataFrame()
    if os.path.exists(local):
        print("File exists, reading locally")
        df = pd.read_csv(local)
    else:
        df = pd.read_csv(url_file)
        df.to_csv(local)
    return df

# Python
# set local or remote file source
file = 'example_wp_log_peyton_manning.csv'
url_file = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'

# read file
df = read_file(file,url_file)

# print head of dataframe
print(df.head())

# plot data
sns.regplot(data=df, x=np.arange(len(df)),  y='y',lowess=True,scatter=True,marker=".",label="Data")
plt.legend()
plt.show()

# make model
