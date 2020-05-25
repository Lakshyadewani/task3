#!/usr/bin/env python
# coding: utf-8
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
dataset=load_digits()
X_train, X_test, y_train, y_test = train_test_split(dataset.data,dataset.target, test_size=0.30, random_state=42)
model = KNeighborsClassifier()
model.fit(X_train,y_train)
pred=model.predict(X_test)
accurate=model.score(X_test,y_test) 
print(accurate)
file1=open("sk_accuracy.txt","w")
file1.write(str(accurate*100))
file1.close()