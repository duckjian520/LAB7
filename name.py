from itertools import count
from tkinter.tix import COLUMN
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer() 
# print(cancer.DESCR) #Print the data set description

print(len(cancer['feature_names']),"\n")

data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
df=pd.DataFrame(data,columns=columns)
print(df,"\n")

counts=df.target.value_counts()
counts.index="malignant benign".split()
print(counts,"\n")

x=df[df.columns[:-1]]
y=df.target
print(x.shape,y.shape,"\n")

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=426,test_size=143)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

model=KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)
knn=model
print(knn,"\n")

a=df.mean()[:-1].values.reshape(1,-1)
print(knn.predict(a),"\n")

predictions=knn.predict(x_test)
print("no cancer: ",len(predictions[predictions==0]),"\n")
print("cancer: ",len(predictions[predictions==1]),"\n")

print(knn.score(x_test,y_test),"\n")