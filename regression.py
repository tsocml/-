from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#补数据函数
def timeseries_to_supervised(data, lag=1):
  df = DataFrame(data)
  columns = [df.shift(i) for i in range(1, lag+1)]
  columns.append(df)
  df = concat(columns, axis=1)
  df.fillna(0, inplace=True)
  return df
#去趋势，差分函数
def difference(dataset, interval=1):
  diff = list()
  for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
  return Series(diff)
#反差分
def inverse_difference(history,yhat,inverse):
    return yhat + history[-inverse]
#归一化，简而言之，归一化的目的就是使得预处理的数据被限定在一定的范围内（比如[0,1]或者[-1,1]），从而消除奇异样本数据导致的不良影响。
def scale(train, test):
  #fit用于计算训练数据的均值和方差， 后面就会用均值和方差来转换训练数据
  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaler = scaler.fit(train)
  # 正态分布
  train = train.reshape(train.shape[0], train.shape[1])
  #将数据转换为正态分布，梯度下降
  train_scaled = scaler.transform(train)
  # 同理
  test = test.reshape(test.shape[0], test.shape[1])
  test_scaled = scaler.transform(test)
  return scaler, train_scaled, test_scaled
  #还原数据
def invert_scale(scaler, X, value):
  new_row = [x for x in X] + [value]
  array = np.array(new_row)
  array = array.reshape(1, len(array))
  inverted = scaler.inverse_transform(array)
  return inverted[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
  X, y = train[:, 0:-1], train[:, -1]
  X = X.reshape(X.shape[0], 1, X.shape[1])
  model = Sequential()
  model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  #训练次数
  for i in range(nb_epoch):
      model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
      model.reset_states()#重置模型状态
  return model
  #预测
def forecast_lstm(model, batch_size, X):
  X = X.reshape(1, 1, len(X))
  yhat = model.predict(X, batch_size=batch_size)
  return yhat[0,0]
series = read_csv('sales-of-shampoo-over-a-three-ye.csv',header=0,parse_dates=[0],index_col=0,encoding='utf-8',squeeze=True)
#series.plot()
#plt.show()
raw_value = series.values
#差分，减趋势
differenced = difference(raw_value,1)
#补数据
supervised = timeseries_to_supervised(differenced,1)
supervised_value = supervised.values
#训练数据和检验数据
train, test = supervised_value[0:-12], supervised_value[-12:]
scaler, train_scaled, test_scaled = scale(train, test)
#训练模型
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
#预测
lstm_model.predict(train_reshaped, batch_size=1)
#反差分。。还原原本数据
inverted = list()
for i in range(len(differenced)):
  value = inverse_difference(series, differenced[i], len(series)-i)
  inverted.append(value)
inverted = Series(inverted)
print(inverted)
print(series)
#分出方差与偏倚
history = [x for x in train]
#print(history)
#预言，魔法
predictions = list()
for i in range(len(test_scaled)):
  # 估计值正态分布
  X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
  yhat = forecast_lstm(lstm_model, 1, X)
  # 还原正态分布
  yhat = invert_scale(scaler, X, yhat)
  # 还原差分
  yhat = inverse_difference(raw_value, yhat, len(test_scaled) + 1 - i)
  # store forecast
  predictions.append(yhat)
  expected = raw_value[len(train) + i + 1]
  #每个月份对应的预测值和真实值
  print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
# 报告绩效
rmse = sqrt(mean_squared_error(raw_value[-12:], predictions))
print('RMSE: %.3f' % rmse)
# 可视化
plt.plot(raw_value[-12:])
plt.plot(predictions)
plt.show()
