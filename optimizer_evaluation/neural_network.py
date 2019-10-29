from keras.layers import Dense
from keras.models import Sequential

class NeuralNetworks:
    """Key value store for neural networks, 'boston' for example is a sequential neural network for the boston dataset
    """
    _regression_NN_boston = Sequential()
    _regression_NN_boston .add(Dense(units=160, activation='relu', input_shape=(13,)))
    _regression_NN_boston .add(Dense(units=64, activation='linear'))
    _regression_NN_boston .add(Dense(units=1, activation='linear'))

    neural_networks = {'boston': _regression_NN_boston}

    def __init__(self):
        pass

    def add(self,name,  neural_network):
        """add new neural nerwork to the key value store 
        
        Arguments:
            name {string} -- the name of the neural network
            neural_network {keras.models} -- a uncompiled keras neural network 

        Example: 
            _regression_NN = Sequential()
            _regression_NN .add(Dense(units=160, activation='relu', input_shape=(13,)))
            _regression_NN .add(Dense(units=64, activation='linear'))
            _regression_NN .add(Dense(units=1, activation='linear'))

            nns = NeuralNetworks()
            nns.add(_regression_NN)
        """
        NeuralNetworks.neural_networks[name] = neural_network

    def compile(self, name, loss, optimizer, metrics):
        """compiles the neural network with the given name
        
        Arguments:
            name {string} -- name of the neural network
            loss {string} -- name of keras objective function, see also keras compile
            optimizer {string} -- name of keras optimizer, see also keras compile
            metrics {string[]} -- name of keras metrics, see also keras compile
        """
        NeuralNetworks.neural_networks[name].compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    def get(self, name):
        return NeuralNetworks.neural_networks[name]

