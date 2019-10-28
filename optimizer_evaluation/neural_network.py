from keras.layers import Dense
from keras.models import Sequential

class NeuralNetworks:
    _regression_NN = Sequential()
    _regression_NN .add(Dense(units=160, activation='relu', input_shape=(13,)))
    _regression_NN .add(Dense(units=64, activation='linear'))
    _regression_NN .add(Dense(units=1, activation='linear'))

    neural_networks = {'regression': _regression_NN}

    def __init__(self):
        pass

    def add(self,name,  neural_network):
        NeuralNetworks.neural_networks[name] = neural_network

    def compile(self, name, loss, optimizer, metrics):
        NeuralNetworks.neural_networks[name].compile(loss=loss, optimizer=optimizer, metrics=[metrics])