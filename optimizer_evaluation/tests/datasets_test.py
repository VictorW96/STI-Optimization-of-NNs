from optimizer_evaluation.datasets import DataSets
import pandas as pd
import optimizer_evaluation
import optimizer_evaluation

datasets = DataSets()


def test_add():
    X = pd.DataFrame([[1,2],[3,4]], columns=['1','2'])
    y = pd.DataFrame([1,2], columns=['1'])
    datasets.add(name='test',X=X,y=y,type_cr='regression')


def test_get_train_test():
    X_train_boston, X_test_boston, y_train_boston, y_test_boston = datasets.get_train_test('boston')


def test_iter():
    for d in datasets:
        print(d)


def test_get():
    test = datasets.get('boston')
    assert type(test) is optimizer_evaluation.featuretarget.FeatureTargetData