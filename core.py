import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import statsmodels.api as sm
import keras

df = pd.read_csv('data.csv')
print(df.head())
print(df.columns)

# sb.distplot(df['Cumulative Confirmed'])
# plt.show()
# sb.pairplot(df, hue="Region", diag_kind="kde", kind="scatter", palette="husl")
# plt.show()
print(df["day"].min())
print(df.isnull().sum())
grpd = df.groupby("day")
print(grpd.day)
y = df.set_index('day')
print(y.head())
print(y.index)
# y = y['Cumulative Deaths'].resample('MS').mean()
y.plot(figsize=(15, 6))
# plt.show()


del y['Country']
del y['Country Name']
del y['Region']
print(y)
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

train_size = int(len(y) * 0.8)
test_size = len(y) - train_size
train, test = y.iloc[0:train_size], y.iloc[train_size:len(y)]
print(len(train), len(test))


import numpy as np

time_steps = 10

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train, time_steps)
X_test, y_test = create_dataset(test, test, time_steps)

print(X_train.shape, y_train.shape)

model = keras.Sequential()
model.add(keras.layers.LSTM(
  units=128,
  input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dense(units=4))
model.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam(0.001)
)



history = model.fit(
    X_train, y_train,
    epochs=1,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)

y_pred = model.predict(X_test)
plt.plot(y_pred)
plt.show()
print(X_test)
