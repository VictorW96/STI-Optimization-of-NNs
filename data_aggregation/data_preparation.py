import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as datasets
from keras.layers import Dense
from keras.models import Sequential

data_boston = datasets.load_boston()
X_boston = pd.DataFrame(data = data_boston.data, columns=data_boston.feature_names)
y_boston = pd.DataFrame(data= data_boston.target, columns = ['price'])

data_breast_cancer = datasets.load_breast_cancer()
X_breast_cancer = pd.DataFrame(data = data_breast_cancer.data, columns=data_breast_cancer.feature_names)
y_breast_cancer = pd.DataFrame(data= data_breast_cancer.target, columns = ['class'])

# Boston House data preparation

# feature and target table
X_y_boston = pd.concat(objs=[X_boston,y_boston] , axis= 1)

# See Correlation of feature and target
boston_cor = X_y_boston.corr()
sns.heatmap(boston_cor, annot=True,  cmap=plt.cm.Reds)
plt.show()

# Neural Network
boston_NN = Sequential()
boston_NN.add(Dense(units=64, activation='relu', input_dim=(13,)))
boston_NN.add(Dense(units=64, activation='softmax'))
boston_NN.add(Dense(units=1, activation='linear'))

boston_NN.compile(loss='mse',
                  optimizer='sgd',
                  metrics=['mae'])
                  
