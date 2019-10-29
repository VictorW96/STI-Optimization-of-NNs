from optimizer_evaluation.neural_network import NeuralNetworks
from keras.models import Sequential
from keras.layers import Dense
import keras
import optimizer_evaluation

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

nns = NeuralNetworks()


def test_add():
    nns.add('test',model,'classification')


def test_iter():
    for nn in nns:
        print(nn)


def test_get():
    test = nns.get('boston')
    assert type(test) is keras.models.Sequential