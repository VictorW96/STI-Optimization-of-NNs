from optimizer_evaluation.tester import Data_Tester
from optimizer_evaluation.datasets import DataSets
from optimizer_evaluation.neural_network import NeuralNetworks

class ADAMTester(Data_Tester):

    optimizer = 'adam'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_all(self):
        for name in self.datasets:

            X_train, X_test, y_train, y_test = self.datasets.get_train_test(name)
            nn = self.neural_networks.get(name)

            if self.datasets.get(name) == 'regression':
                loss = 'mse'
                metrics = ['mae']
            else:
                loss = 'binary_crossentropy'
                metrics=['accuracy']

            nn.compile(loss=loss,
                       optimizer= ADAMTester.optimizer,
                       metrics=metrics)

            nn.fit(X_train.values,y_train.values,epochs=20,batch_size=1)
            predict = nn.predict(X_test.values, batch_size=1)