import pandas as pd
import sklearn.datasets as skdatasets
from optimizer_evaluation.featuretarget import FeatureTargetData
from sklearn.model_selection import train_test_split

class DataSets:
    """DataSets is a key value save for Data Sets. Standard Sets are
    'boston',type:regression and 'breast_cancer',type:classification """

    _data_boston = skdatasets.load_boston()
    _X_boston = pd.DataFrame(data = _data_boston.data, columns=_data_boston.feature_names)
    _y_boston = pd.DataFrame(data= _data_boston.target, columns = ['price'])

    _data_breast_cancer = skdatasets.load_breast_cancer()
    _X_breast_cancer = pd.DataFrame(data = _data_breast_cancer.data, columns=_data_breast_cancer.feature_names)
    _y_breast_cancer = pd.DataFrame(data= _data_breast_cancer.target, columns = ['class'])

    datasets = {'boston' : FeatureTargetData(_X_boston,_y_boston,'regression'),
                 'breast_cancer' : FeatureTargetData(_X_breast_cancer,_y_breast_cancer,'classification')}

    def __init__(self):
        pass

    def add(self, name, X, y, type_cr):
        """
        add new dataframe to the dataset class 
        
        Arguments:
            name {String} -- name of dataframe 
            X {pandas.DataFrame} -- features as nxn pandas DataFrame
            y {pandas.DataFrame} -- target as nx1 pandas Dataframe
            type {string} -- [either regression or classification ]
        """
        DataSets.datasets[name] = FeatureTargetData(X,y,type_cr)

    def get_train_test(self,name):
        """return train_test_split of sklearn to the corresponding dataset
        
        Arguments:
            name {string} -- name of dataset  
        """
        return train_test_split(DataSets.datasets[name].X,DataSets.datasets[name].y,test_size=0.2)

    def get(self, name):
        return DataSets.datasets[name]

    def __iter__(self):
        return iter(DataSets.datasets)

    def next(self):
        return next(DataSets.datasets)
