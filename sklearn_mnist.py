#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.neighbors import KNeighborsClassifier


# In[21]:


dataset=load_digits()


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(dataset.data,dataset.target, test_size=0.30, random_state=42)


# In[16]:


model = KNeighborsClassifier()


# In[17]:


model.fit(X_train,y_train)


# In[18]:


pred=model.predict(X_test)


# In[19]:


accurate=model.score(X_test,y_test) 


# In[22]:


print(accurate)


# In[23]:


file1=open("sk_accuracy.txt","w")


# In[26]:


file1.write(str(accurate*100))
file1.close()


# In[ ]:




