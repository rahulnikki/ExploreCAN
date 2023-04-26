#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import category_encoders as ce

df = pd.read_csv('Restuarants.csv')
df.head(5)


# In[26]:


df.info()


# In[20]:


encoder = ce.OrdinalEncoder(cols=['Cuisine Type'])
encoder.fit(df)
df = encoder.transform(df)
df


# In[21]:


df.info


# In[24]:


df.describe()


# In[25]:


df['Cuisine Type'].unique()


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np


# In[7]:


df_num = df[['Cuisine Type','Average Rating (out of 5)', 'Number of Reviews']]
df_num


# In[8]:


X = df_num.drop('Average Rating (out of 5)', axis=1)
y = df_num['Average Rating (out of 5)']


# In[9]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)  

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True) 

rf.fit(X_train, y_train) 

rf.score(X_train, y_train)


# In[10]:


rf.score(X_val, y_val)


# In[11]:


rf.oob_score_


# In[12]:


train_r2 = []
train_mae = []
val_r2 = []
val_mae = []
oob_scores = []

for i in range(10):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True) 
    rf.fit(X_train, y_train)
    train_preds = rf.predict(X_train)
    val_preds = rf.predict(X_val)
    train_r2.append(round(r2_score(y_train, train_preds), 2))
    val_r2.append(round(r2_score(y_val, val_preds), 2))
    train_mae.append(round(mean_absolute_error(y_train, train_preds), 0))
    val_mae.append(round(mean_absolute_error(y_val, val_preds), 0))
    oob_scores.append(rf.oob_score_)


# In[13]:


print("Train r2 scores: \n", train_r2)
print("")
print("Validation r2 scores: \n", val_r2)
print("")
print("Train MAE scores: \n", train_mae)
print("")
print("Validation MAE scores: \n", val_mae)
print("")
print("Out-of-bag scores: \n", oob_scores)


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=41)


# In[19]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 
from math import sqrt
rms_val = []
for K in range(10):
    K = K+1
    model = KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train, y_train)
    pred=model.predict(X_test)
    error = sqrt(mean_squared_error(y_test,pred))
    rms_val.append(error)
    print('RMS value for k= ' , K , 'is:', error)


# In[ ]:




