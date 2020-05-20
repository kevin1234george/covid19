# importing required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

data = pd.read_csv('data.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)

print(data.head())
del data['Country']
del data['Country Name']
# del data['Region']

print(data.head())
print(data.index)
print(data.head())
print(data.index)
data['day'] = pd.to_datetime(data['day'])

data.set_index("day", inplace=True)

# ts = data['Deaths']
ts = data
print(ts.head(10))

# sb.pairplot(data, diag_kind="kde", kind="scatter", palette="husl")


# polyfit
# sb.regplot(x="Confirmed", y="Deaths", data=data,
#                )

k = ts["Region"] == "EMRO"
print(k)
test = ts[k]

print(test)


# plt.plot(ts["Cumulative Deaths"])
# sb.lmplot(data=test, hue="Region", x="Confirmed", y="Deaths", col="Region", col_wrap=2, height=3)
sb.lmplot(data=test, hue="Region", x="Confirmed", y="Deaths")

# plt.show()







import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

# x = numpy.random.normal(3, 1, 100)
# y = numpy.random.normal(150, 40, 100) / x

train_x = data["Cumulative Confirmed"]
train_y = data["Cumulative Deaths"]

# test_x = x[80:]
# test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 1703500, 1703200)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()