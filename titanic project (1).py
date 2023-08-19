#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


titanic_data=pd.read_csv('downloads/tested.csv')


# In[5]:


titanic_data


# In[6]:


import seaborn as sns
sns.heatmap(titanic_data.corr(),cmap="YlGnBu")
plt.show()


# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2)
for train_indices, test_indices in split.split(titanic_data,titanic_data[["Survived","Pclass","Sex"]]):
    strat_train_set=titanic_data.loc[train_indices]
    srat_test_set=titanic_data.loc[test_indices]                                                                                                                            


# In[18]:


plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.show()


# In[15]:


strat_train_set.info()


# In[19]:


from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer

class AgeIputer(BaseEstimator,TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        imputer=SimpleImputer(strategy="mean")
        X['Age']=imputer.fit_transform(X[['Age']])
        return X
        


# In[20]:


from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        encoder=OneHotEncoder()
        matrix=encoder.fit_transform(X[['Embarked']]).toarray()
        
        column_names=["C","S","Q","N"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]]=matrix.T[i]
        matrix=encoder.fit_transform(X[['Sex']]).toarray()
        
        column_names=["Female","Male"]
        
        for i in range(len(matrix.T)):
            X[column_name[i]] = matrix.T[i]
        return X


# In[21]:


class FeatureDropper(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"], axis=1,errors="ignore")


# In[39]:


from sklearn.pipeline import Pipeline

pipeline= Pipeline([("ageimputer",()),
                 ("featureencoder", FeatureEncoder()),
                 ("featureddropper", FeatureDropper())])


# In[41]:


strat_train_set


# In[42]:


strat_train_set.info()


# In[73]:


sns.countplot(x='Survived', data=titanic_data)


# In[75]:


sns.countplot(x='Survived',hue='Sex',data=titanic_data)


# In[76]:


sns.catplot(x='Pclass',hue='Survived',kind='count',data=titanic_data)


# In[77]:


plt.figure(figsize=(10,10))
plt.subplot(3,2,6)
sns.boxplot(x='Pclass',y='Age',data=titanic_data)


# In[83]:


def add_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return titanic_data[titanic_data['Pclass']==1]['Age'].median()
        elif Pclass==2:
            return titanic_data[titanic_data['Pclass']==2]['Age'].median()
        elif Pclass==3:
            return titanci_data[titanic_data['Pclass']==3]['Age'].median()
    else:
        return Age


# In[84]:


titanic_data.head()


# In[85]:


x_data=titanic_data.drop('Survived',axis=1)
y_data=titanic_data['Survived']


# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2, random_state=0, stratify=y_data)


# In[90]:


from sklearn.linear_model import LogisticRegression


# In[91]:


model = LogisticRegression()


# In[103]:


from sklearn.metrics import classification_report


# In[ ]:


model.fit(x_test_data, y_titanic_data)
predictions = model.predict(x_titanic_data)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test_data, predictions))


# In[ ]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test_data, predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_data, predictions)


# In[ ]:


import seaborn as sns

sns.heatmap(cf_matrix, annot=True)


# In[ ]:


prediction=model.predict(test_data)
test=pd.read_csv("../input/titanic/tested.csv")
prediction


# In[ ]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": prediction})
submission.to_csv('submission.csv', index=False)
submission

