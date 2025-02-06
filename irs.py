import pandas as pd
dataset=pd.read_csv(r'C:\\Users\\gaurav raikwar\\Downloads\\Iris.csv')
dataset
dataset['Species'].unique()
dataset['Species'].replace({'Iris-setosa':'1','Iris-versicolor':'2','Iris-virginica':'3'},inplace=True)
dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],dataset['Species'],test_size=0.2)
len(x_train)
len(x_test)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.predict(x_test)