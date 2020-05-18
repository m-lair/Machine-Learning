#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wine = pd.read_csv('winequality-red.csv', sep=';')


# In[3]:


wine.head()


# In[4]:


wine.info()


# In[5]:


bins = (2,6.5,8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
wine['quality'].unique()


# In[6]:


label_quality = LabelEncoder()


# In[7]:


wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[11]:


wine.head(10)


# In[12]:


wine['quality'].value_counts()


# In[13]:


sns.countplot(wine['quality'])


# In[14]:


X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[17]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[19]:


#random forest classifier


# In[21]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[24]:


print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# In[27]:


clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)


# In[30]:


print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))


# In[31]:


mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)

    
    


# In[32]:


print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# In[33]:


from sklearn.metrics import accuracy_score


# In[ ]:




