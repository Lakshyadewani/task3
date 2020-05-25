#!/usr/bin/env python
# coding: utf-8
#firstly import all the pakages which is used in mnist CNN
from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
#load the mnist dataset
dataset = mnist.load_data('mymnist.db')
# dataset length is 2 so divide into train and test 
train , test = dataset
# same as for train and test part divide into X and y
X_train , y_train = train
X_test , y_test = test
# now we have to flattern the image form 28*28 to 784 or convert our image as 1d  
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
# change our image datatype to float
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')
# one hot encoading outputs
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
# creating the model and add layers  
model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
# output layer
model.add(Dense(units=10, activation='softmax'))
# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# fitting our X and y in model
h=model.fit(X_train, y_train_cat, epochs=10)
accurate = model.evaluate(X_test,y_test_cat,verbose=0)
# to print the accuracy of the model
print("accuracy : %.2f%%"% (accurate[1]*100))
# saving the model
model.save("mnist_model.h5")
# put our accuracy as a result in the file 
file1=open("accuracy.txt","w")
file1.write(str(accurate[1]*100))
file1.close()