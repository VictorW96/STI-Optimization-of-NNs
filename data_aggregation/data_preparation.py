import sklearn.datasets as datasets
import pandas as pd

data_boston = datasets.load_boston()
df_boston = pd.DataFrame(data = data_boston.data, columns=data_boston.feature_names)
print(df_boston.head())

data_breast_cancer = datasets.load_breast_cancer()
df_breast_cancer = pd.DataFrame(data = data_breast_cancer.data, columns=data_breast_cancer.feature_names)
print(df_breast_cancer.head())

