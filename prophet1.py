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

def doin_prophet(df:pd.DataFrame,period = 365) -> pd.DataFrame:
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    plt.show(block=False)
    m.plot_components(forecast)
    plt.show(block=True)
    return forecast,m

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
df_prophet, m = doin_prophet(df)

file1 = "PJME_hourly.csv"
df1 = read_file(file1,"")
print(df1.head())

# make model
df1_prophet, m1 = doin_prophet(df1)

# plot data
sns.regplot(data=df, x=np.arange(len(df)),  y='y',lowess=True,scatter=True,marker=".",label="Data")
plt.legend()
plt.show()

