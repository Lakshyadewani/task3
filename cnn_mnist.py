#!/usr/bin/env python
# coding: utf-8

# In[1]:


#firstly import all the pakages which is used in mnist CNN
from keras.datasets import mnist 


# In[2]:


from keras.models import Sequential


# In[3]:


from keras.layers import Dense


# In[4]:


from keras.utils.np_utils import to_categorical


# In[5]:


#load the mnist dataset
dataset = mnist.load_data('mymnist.db')


# In[6]:


# dataset length is 2 so divide into train and test 
train , test = dataset


# In[7]:


# same as for train and test part divide into X and y
X_train , y_train = train


# In[8]:


X_test , y_test = test


# In[9]:


# now we have to flattern the image form 28*28 to 784 or convert our image as 1d  
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[10]:


# change our image datatype to float
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[11]:


# one hot encoading outputs
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[12]:


# creating the model and add layers  
model = Sequential()


# In[13]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))


# In[14]:


model.add(Dense(units=256, activation='relu'))


# In[15]:


model.add(Dense(units=128, activation='relu'))


# In[16]:


model.add(Dense(units=32, activation='relu'))


# In[49]:


# output layer
model.add(Dense(units=10, activation='softmax'))


# In[18]:


# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[24]:


# fitting our X and y in model
h=model.fit(X_train, y_train_cat, epochs=10)


# In[32]:


accurate = model.evaluate(X_test,y_test_cat,verbose=0)


# In[36]:


# to print the accuracy of the model
print("accuracy : %.2f%%"% (accurate[1]*100))


# In[37]:


# saving the model
model.save("mnist_model.h5")


# In[51]:


# put our accuracy as a result in the file 
file1=open("accuracy.txt","w")


# In[52]:


file1.write(str(accurate[1]*100))
file1.close()


# In[47]:





# In[ ]:




