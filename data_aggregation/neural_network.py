from keras.layers import Dense
from keras.models import Sequential

class NeuralNetworks:
    _regression_NN = Sequential()
    _regression_NN .add(Dense(units=160, activation='relu', input_shape=(13,)))
    _regression_NN .add(Dense(units=64, activation='linear'))
    _regression_NN .add(Dense(units=1, activation='linear'))

    _neural_networks = {'regression': _regression_NN}

    def __init__(self):
        pass

    def add(self, neural_network):
        pass

    def fit(self, name):
        pass

    def 
