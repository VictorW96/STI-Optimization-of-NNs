from optimizer_evaluation.tester import Data_Tester
from optimizer_evaluation.datasets import DataSets
from optimizer_evaluation.neural_network import NeuralNetworks

class ADAM_tester(Data_Tester):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self):
        for name in self.datasets:
            data = self.datasets.get(name)