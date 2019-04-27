import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pytrade.backtesting.backtest import GoogleFinanceBacktest

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
codes = ["BBAS3"]
feed = GoogleFinanceBacktest(instruments=codes, initialCash=10000, fromYear=2008, toYear=2019, debugMode=False,
                                     csvStorage="./googlefinance", filterInvalidRows=False).getFeed()
feed.loadAll()

# code= codes[0]
# closevalues = feed.getDataSeries(instrument=code).getCloseDataSeries()
# data = pd.DataFrame(data={'Close': np.array(closevalues), 'Date': closevalues.getDateTimes()})
# data.set_index('Date', drop=True, inplace=True)
# print(data.describe())





scaler = MinMaxScaler()
DataScaled = scaler.fit_transform(data)

TrainLen = int(len(DataScaled) * 0.70)
TestLen = len(DataScaled) - TrainLen
TrainData = DataScaled[0:TrainLen,:]
TestData = DataScaled[TrainLen:len(DataScaled),:]
print(len(TrainData), len(TestData))

def DatasetCreation(dataset, TimeStep=1):
   DataX, DataY = [], []
   for i in range(len(dataset) - TimeStep):
         a = dataset[i:(i+ TimeStep), 0]
         DataX.append(a)
         DataY.append(dataset[i + TimeStep, 0])
   return np.array(DataX), np.array(DataY)

TimeStep = 120
TrainX, TrainY = DatasetCreation(TrainData, TimeStep)
TestX, TestY = DatasetCreation(TestData, TimeStep)

TrainX = np.reshape(TrainX, (TrainX.shape[0], TimeStep, 1))
TestX = np.reshape(TestX, (TestX.shape[0], TimeStep, 1))

model = Sequential()
model.add(LSTM(1024, input_shape=(TimeStep, 1)))
model.add(Dense(1024, activation='linear'))
model.add(Dense(512, activation='linear'))
model.add(Dense(256, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
model.fit(TrainX, TrainY, epochs=30, batch_size=10, verbose=1)
model.summary()
score = model.evaluate(TrainX, TrainY, verbose=0)
print('Keras Model Loss = ',score[0])
print('Keras Model mse = ',score[1])

TrainPred = model.predict(TrainX)
TestPred = model.predict(TestX)
TrainPred = scaler.inverse_transform(TrainPred)
TrainY = scaler.inverse_transform([TrainY])
TestPred = scaler.inverse_transform(TestPred)
TestY = scaler.inverse_transform([TestY])


TrainPredictPlot = np.empty_like(DataScaled)
TrainPredictPlot[:, :] = np.nan
TrainPredictPlot[TimeStep:len(TrainPred)+TimeStep, :] = TrainPred
TestPredictPlot = np.empty_like(DataScaled)
TestPredictPlot[:, :] = np.nan
TestPredictPlot[TimeStep+len(TrainPred)+TimeStep:, :] = TestPred
plt.plot(scaler.inverse_transform(DataScaled), color='blue', marker='o', markersize=2)
plt.plot(TrainPredictPlot, color='green', marker='o', markersize=2)
plt.plot(TestPredictPlot, color='yellow', marker='o', markersize=2)
plt.legend()
plt.show()


trainY_up = TrainY > scaler.inverse_transform(TrainX[:, -1, 0])
trainY_up = trainY_up[0]
trainPred_up = model.predict(TrainX) > TrainX[:, -1, 0]
trainPred_up = trainPred_up[0]
cm = confusion_matrix(trainPred_up, trainY_up)
accuracy = float(cm[0][0] + cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Model accuracy for train %s" % (accuracy))

testY_up = TestY > scaler.inverse_transform(TestX[:, -1, 0])
testY_up = testY_up[0]
testPred_up = model.predict(TestX) > TestX[:, -1, 0]
testPred_up = testPred_up[0]
cm = confusion_matrix(testPred_up, testY_up)
accuracy = float(cm[0][0] + cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Model accuracy for test = %s" % (accuracy))












########################3
def DatasetCreation(dataset, TimeStep=1):
   DataX, DataY = [], []
   for i in range(len(dataset)- TimeStep):
         DataX.append(dataset[i:(i+ TimeStep), 0])
         DataY.append(1 if dataset[i + TimeStep, 0]>dataset[i + TimeStep - 1, 0] else 0)
   return np.array(DataX), np.array(DataY)

TimeStep = 180
TrainX, TrainY = DatasetCreation(TrainData, TimeStep)
TestX, TestY = DatasetCreation(TestData, TimeStep)

TrainX = np.reshape(TrainX, (TrainX.shape[0], TimeStep, 1))
TestX = np.reshape(TestX, (TestX.shape[0], TimeStep, 1))


model = Sequential()
model.add(LSTM(256, input_shape=(TimeStep, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(TrainX, TrainY, epochs=100, batch_size=1, verbose=1)
model.summary()

score = model.evaluate(TrainX, TrainY, verbose=0)
print('Keras Model Loss for Train = ',score[0])
print('Keras Model accuracy for Train = ',score[1])

score = model.evaluate(TestX, TestY, verbose=0)
print('Keras Model Loss for Test = ',score[0])
print('Keras Model accuracy for Test = ',score[1])

TrainPred = model.predict(TrainX)>0.5
TestPred = model.predict(TestX)>0.5

cm = confusion_matrix(TestPred, TestY==1)
print(cm)