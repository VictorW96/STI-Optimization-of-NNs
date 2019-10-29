from keras.layers import Dense
from keras.models import Sequential
import optimizer_evaluation

class NeuralNetworks:
    """Key value store for neural networks, 'boston' for example is a sequential neural network for the boston dataset
    """
    _regression_NN_boston = Sequential()
    _regression_NN_boston.add(Dense(units=160, activation='relu', input_shape=(13,)))
    _regression_NN_boston.add(Dense(units=64, activation='linear'))
    _regression_NN_boston.add(Dense(units=1, activation='linear'))

    _classification_NN_breast_cancer = Sequential()
    _classification_NN_breast_cancer.add(Dense(40, input_shape=(30,), activation='relu'))
    _classification_NN_breast_cancer.add(Dense(20, activation='relu'))
    _classification_NN_breast_cancer.add(Dense(1, activation='sigmoid'))

    def __init__(self):
        self.neural_networks = {'boston': NeuralNetworks._regression_NN_boston,
                                'breast_cancer':NeuralNetworks._classification_NN_breast_cancer}

    def add(self,name,  neural_network, type_rc):
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
        self.neural_networks[name] = (neural_network, type_rc)

    def get(self, name):
        return self.neural_networks[name]

    def __iter__(self):
        return iter(self.neural_networks)
        
    def __next__(self):
        return next(self.neural_networks)

