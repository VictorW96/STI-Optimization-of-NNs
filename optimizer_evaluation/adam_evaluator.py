import optimizer_evaluation
from optimizer_evaluation.data_evaluator import Data_Evaluator
from optimizer_evaluation.datasets import DataSets
from optimizer_evaluation.neural_network import NeuralNetworks
import matplotlib.pyplot as plt

##

class ADAMEvaluator(Data_Evaluator):


    optimizer = 'adam'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_all(self):
        for name in self.datasets:

            X_train, X_test, y_train, y_test = self.datasets.get_train_test(name)
            nn = self.neural_networks.get(name)

            if self.datasets.get(name).type == 'regression':
                loss = 'mse'
                metrics = ['mae']
            else:
                loss = 'binary_crossentropy'
                metrics=['accuracy']

            nn.compile(loss=loss,
                       optimizer= ADAMEvaluator.optimizer,
                       metrics=metrics)

            nn.fit(X_train.values,y_train.values,epochs=20,batch_size=1)
            predict = nn.predict(X_test.values, batch_size=1)

            
            fig, ax = plt.subplots(1,1)
            ax.plot(predict, color='green', marker='o', linestyle= 'dotted')
            ax.plot(y_test.values, color='blue', marker='o', linestyle= 'dotted')
            plt.savefig("docs/boston/bost_test.png")