import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


risoe_ds = xr.open_dataset('C:/Users/45527/Downloads/Data/Risoe/risoe_m_all.nc')
borglum_ds = xr.open_dataset('C:/Users/45527/Downloads/Data/Borglum/borglum_all.nc')


risoe_ds = risoe_ds[['ws77', 'wd77']]
borglum_ds = borglum_ds[['ws77', 'wd77']]


risoe_ds = risoe_ds.dropna(dim='time')
borglum_ds = borglum_ds.dropna(dim='time')


risoe_ds['wd77'] = risoe_ds['wd77'] % 360
borglum_ds['wd77'] = borglum_ds['wd77'] % 360


risoe_ds['time'] = risoe_ds.indexes['time'].to_datetimeindex() - pd.Timedelta(hours=1)
borglum_ds['time'] = borglum_ds.indexes['time'].to_datetimeindex() - pd.Timedelta(hours=1)


risoe_ds = risoe_ds.resample(time='1H').mean()
borglum_ds = borglum_ds.resample(time='1H').mean()


common_times = risoe_ds.indexes['time'].intersection(borglum_ds.indexes['time'])

risoe_common = risoe_ds.sel(time=common_times)
borglum_common = borglum_ds.sel(time=common_times)


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(risoe_common.indexes['time'], risoe_common['ws77'], label='Risoe')
plt.plot(borglum_common.indexes['time'], borglum_common['ws77'], label='Borglum')
plt.title('Wind Speed at 77m')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(risoe_common.indexes['time'], risoe_common['wd77'], label='Risoe')
plt.plot(borglum_common.indexes['time'], borglum_common['wd77'], label='Borglum')
plt.title('Wind Direction at 77m')
plt.legend()

plt.tight_layout()
plt.show()

c_risoe, loc_risoe, scale_risoe = weibull_min.fit(risoe_common['ws77'], floc=0)
c_borglum, loc_borglum, scale_borglum = weibull_min.fit(borglum_common['ws77'], floc=0)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = risoe_common[['ws77', 'wd77']].to_pandas().values  
y = borglum_common['ws77'].to_pandas().values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
