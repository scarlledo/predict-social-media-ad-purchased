#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data=pd.read_csv('data.csv')


# In[4]:


data.isnull().sum()


# In[5]:


data.head()


# In[10]:


data = data.drop('User ID', axis=1)


# In[11]:


data.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


le= LabelEncoder()


# In[15]:


data['Gender']=le.fit_transform(data['Gender'])


# In[16]:


data.head()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


sns.pairplot(data)


# In[20]:


x=data.drop('Purchased',axis=1).values
y=data['Purchased'].values


# In[21]:


x.shape,y.shape


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True)


# In[24]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


log=LogisticRegression()


# In[30]:


log.fit(x_train,y_train)


# In[31]:


pred = log.predict(x_test)


# In[32]:


from sklearn.metrics import accuracy_score


# In[33]:


accuracy_score(y_test,pred)


# In[34]:


plt.plot(pred, '--r')
plt.plot(y_test, '-b')
plt.title('Pur vs Act')
plt.show()


# In[ ]:




