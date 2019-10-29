from optimizer_evaluation.neural_network import NeuralNetworks
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

nns = NeuralNetworks()

def test_add():
    nns.add('test',model)

def test_compile():
    nns.compile('test','binary_crossentropy','adam',['accuracy'])

def test_iter():
    for nn in nns:
        print(nn)