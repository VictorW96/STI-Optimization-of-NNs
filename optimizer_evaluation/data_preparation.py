import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
import optimizer_evaluation

data_boston = datasets.load_boston()
X_boston = pd.DataFrame(data = data_boston.data, columns=data_boston.feature_names)
y_boston = pd.DataFrame(data= data_boston.target, columns = ['price'])

data_breast_cancer = datasets.load_breast_cancer()
X_breast_cancer = pd.DataFrame(data = data_breast_cancer.data, columns=data_breast_cancer.feature_names)
y_breast_cancer = pd.DataFrame(data= data_breast_cancer.target, columns = ['class'])

# Neural Network structure
boston_NN = Sequential()
boston_NN.add(Dense(units=160, activation='relu', input_shape=(13,)))
boston_NN.add(Dense(units=64, activation='linear'))
boston_NN.add(Dense(units=1, activation='linear'))

# NN compile
boston_NN.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae'])

# train data fitting
X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(X_boston,y_boston,test_size=0.2)
boston_NN.fit(X_train_boston.values,y_train_boston,epochs=20,batch_size=1)

                  
predict=boston_NN.predict(X_test_boston, batch_size=1)
error = np.abs(predict - y_test_boston.values)

fig, ax = plt.subplots(1,1)
ax.plot(predict, color='green', marker='o', linestyle= 'dotted')
ax.plot(y_test_boston.values, color='blue', marker='o', linestyle= 'dotted')
plt.show()

