import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data.csv')
print(df.head())

df = df.sort_values('day')
print(df.head())

plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), (df['Cumulative Deaths']) / 2.0)
plt.xticks(range(0, df.shape[0], 500), df['day'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid death', fontsize=18)
# plt.show()

del df['Country']
del df['Country Name']
del df['Region']

print(df)

high_prices = df.loc[:, 'Cumulative Deaths'].values
low_prices = df.loc[:, 'Deaths'].values
mid_prices = (high_prices + low_prices) / 2.0

train_data = mid_prices[:110]
test_data = mid_prices[110:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])