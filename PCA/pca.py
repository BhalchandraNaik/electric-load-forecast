import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt




df_power = []
for i in range(1,11):
    df_power_zone = pd.read_csv('../data/Task 15/Task15_W_Zone1_10/Task15_W_Zone'+str(i)+'.csv', 
                                header=0, 
                                usecols=[1, 2], 
                                names=['datetime', 'wf'+str(i)])
    df_power_zone['datetime'] = pd.to_datetime(df_power_zone['datetime'], format='%Y%m%d %H:%M')
    df_power_zone.index = df_power_zone['datetime']
    df_power_zone = df_power_zone.drop(['datetime'], axis=1)
    df_power.append(df_power_zone)
df_power = pd.concat(df_power, axis=1, join='outer')




print(df_power.index[0])
print(df_power.index[-1])
df_power.head()

df_power = df_power.fillna(method='ffill')


from sklearn.decomposition import PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(df_power)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['1','2','3','4','5','6','7','8','9','10'])



principalDf.plot()
