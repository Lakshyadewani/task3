#!/usr/bin/env python
# coding: utf-8

# In[160]:


#firstly import all the pakages which is used in mnist CNN
from keras.datasets import mnist 


# In[161]:


from keras.models import Sequential


# In[162]:


from keras.layers import Dense


# In[163]:


from keras.utils.np_utils import to_categorical


# In[164]:


#load the mnist dataset
dataset = mnist.load_data('mymnist.db')


# In[165]:


# dataset length is 2 so divide into train and test 
train , test = dataset


# In[166]:


# same as for train and test part divide into X and y
X_train , y_train = train


# In[167]:


X_test , y_test = test


# In[168]:


# now we have to flattern the image form 28*28 to 784 or convert our image as 1d  
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[169]:


# change our image datatype to float
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[170]:


# one hot encoading outputs
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[171]:


# creating the model and add layers  
model = Sequential()


# In[172]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))


# In[173]:


model.add(Dense(units=256, activation='relu'))


# In[174]:


model.add(Dense(units=128, activation='relu'))


# In[175]:


model.add(Dense(units=32, activation='relu'))


# In[177]:


# output layer
model.add(Dense(units=10, activation='softmax'))


# In[178]:


# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[179]:


# fitting our X and y in model
h=model.fit(X_train, y_train_cat, epochs=10)


# In[181]:


accurate = model.evaluate(X_test,y_test_cat,verbose=0)


# In[182]:


# to print the accuracy of the model
print("accuracy : %.2f%%"% (accurate[1]*100))


# In[183]:


# saving the model
model.save("mnist_model.h5")


# In[184]:


# put our accuracy as a result in the file 
file1=open("accuracy.txt","w")


# In[185]:


file1.write(str(accurate[1]*100))
file1.close()


# In[204]:


from numpy import loadtxt
from keras.models import load_model


# In[231]:


# load model
model = load_model('mnist_model.h5')


# In[232]:


for layer in model.layers:
    layer.trainable = False


# In[233]:


top_model=model.output


# In[234]:


top_model=Dense(units=16, activation='relu')(top_model)


# In[235]:


top_model=Dense(units=10, activation='softmax')(top_model)


# In[236]:


model.compile(optimizer='adam', loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[237]:


h=model.fit(X_train, y_train_cat, epochs=2)


# In[238]:


accurate = model.evaluate(X_test,y_test_cat,verbose=0)


# In[239]:


print("accuracy : %.2f%%"% (accurate[1]*100))


# In[240]:


model.save("mnist_tweak_model.h5")


# In[241]:


file1=open("accuracy_tweak.txt","w")


# In[242]:


file1.write(str(accurate[1]*100))
file1.close()


# In[ ]:




