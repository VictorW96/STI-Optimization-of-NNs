import sklearn.datasets as datasets
import pandas as pd

data_boston = datasets.load_boston()
X_boston = pd.DataFrame(data = data_boston.data, columns=data_boston.feature_names)
y_boston = pd.DataFrame(data= data_boston.target, columns = ['price'])

data_breast_cancer = datasets.load_breast_cancer()
X_breast_cancer = pd.DataFrame(data = data_breast_cancer.data, columns=data_breast_cancer.feature_names)
y_breast_cancer = pd.DataFrame(data= data_breast_cancer.target, columns = ['class'])