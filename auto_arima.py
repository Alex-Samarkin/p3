# Python
# import modules
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import os

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf



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
from pmdarima.arima import auto_arima

x = df['y'].values

# Best model:  ARIMA(7,1,7)(2,0,2)[12]          
# Total fit time: 1900.167 seconds

# Best model:  ARIMA(5,1,8)(2,0,1)[12]          
# Total fit time: 1049.177 seconds

model = auto_arima(x, start_p=4, start_q=4,
                      test='adf',
                      max_p=8, max_q=8,
                      # must be 12 for monts, 52 weeks, 7 dsyly etc
                      m=12,             
                      d=1,          
                      seasonal=True,   
                      start_P=0, 
                      start_Q=0,
                      D=None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

# print model summary
print(model.summary())

# make forecast
forecast = model.predict(n_periods=240)

model.plot_diagnostics(figsize=(14,10))
plt.show()