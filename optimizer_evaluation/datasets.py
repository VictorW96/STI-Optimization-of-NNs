import pandas as pd
import sklearn.datasets as skdatasets
from optimizer_evaluation.featuretarget import FeatureTargetData
from sklearn.model_selection import train_test_split
import optimizer_evaluation


class DataSets:
    """DataSets is a key value save for Data Sets. Standard Sets are
    'boston',type:regression and 'breast_cancer',type:classification """

    _data_boston = skdatasets.load_boston()
    _X_boston = pd.DataFrame(data=_data_boston.data,
                             columns=_data_boston.feature_names)
    _y_boston = pd.DataFrame(data=_data_boston.target, columns=['price'])

    _data_breast_cancer = skdatasets.load_breast_cancer()
    _X_breast_cancer = pd.DataFrame(
        data=_data_breast_cancer.data, 
        columns=_data_breast_cancer.feature_names)
    _y_breast_cancer = pd.DataFrame(
        data=_data_breast_cancer.target, columns=['class'])

    def __init__(self):
        self.datasets = {'boston': FeatureTargetData(DataSets._X_boston, DataSets._y_boston, 'regression'),
                         'breast_cancer': FeatureTargetData(DataSets._X_breast_cancer, DataSets._y_breast_cancer, 'classification')}

    def add(self, name, X, y, type_cr):
        """
        add new dataframe to the dataset class 

        Arguments:
            name {String} -- name of dataframe 
            X {pandas.DataFrame} -- features as nxn pandas DataFrame
            y {pandas.DataFrame} -- target as nx1 pandas Dataframe
            type {string} -- [either regression or classification ]
        """
        self.datasets[name] = FeatureTargetData(X, y, type_cr)

    def get_train_test(self, name):
        """return train_test_split of sklearn to the corresponding dataset

        Arguments:
            name {string} -- name of dataset  
        """
        return train_test_split(self.datasets[name].X, self.datasets[name].y, test_size=0.2)

    def get(self, name):
        return self.datasets[name]

    def __iter__(self):
        return iter(self.datasets)

    def __next__(self):
        return next(self.datasets)
