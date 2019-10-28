import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from optimizer_evaluation.datasets import DataSets
import pandas as pd

datasets = DataSets()

def test_add():
    X = pd.DataFrame([[1,2],[3,4]], columns=['1','2'])
    y = pd.DataFrame([1,2], columns=['1'])
    datasets.add(name='test',X=X,y=y,type_cr='regression')

def test_get_train_test():
    X_train_boston, X_test_boston, y_train_boston, y_test_boston = datasets.get_train_test('boston')
