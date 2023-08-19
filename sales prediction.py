#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


# In[ ]:


advertising = pd.DataFrame(pd.read_csv("advertising.csv"))
advertising.head()


# In[ ]:


advertising.shape


# In[ ]:


advertising.info()


# In[ ]:


advertising.describe()


# In[ ]:


advertising.isnull().sum()*100/advertising.shape[0]


# In[ ]:


fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
plt.tight_layout()


# In[ ]:


sns.boxplot(advertising['Sales'])
plt.show()


# In[ ]:


sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[ ]:


sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[ ]:


X = advertising['TV']
y = advertising['Sales']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


import statsmodels.api as sm


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()


# In[ ]:


lr.params


# In[ ]:


print(lr.summary())


# In[ ]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[ ]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[ ]:


plt.scatter(X_train,res)
plt.show()


# In[ ]:


X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)


# In[ ]:


y_pred.head()


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


r_squared = r2_score(y_test, y_pred)
r_squared


# In[ ]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:


def conversion(week,days,months,years,list_row):
 
  inp_day = []
  inp_mon = []
  inp_year = []
  inp_week=[]
  inp_hol=[]
  out = []
   week1 = number_to_one_hot(week)
 
  for row in list_row:
        
        d = row[0]
          d_split=d.split('/')
        if d_split[2]==str(year_all[0]):
            continue
            d1,m1,y1 = date_to_enc(d,days,months,years)  containing the one hot encoding of each date,month and year.
        inp_day.append(d1) 
        inp_mon.append(m1) 
        inp_year.append(y1) 
        week2 = week1[row[3]] 
        inp_week.append(week2)
        inp_hol.append([row[2]])
        t1 = row[1] out.append(t1) 
  return inp_day,inp_mon,inp_year,inp_week,inp_hol,out 
inp_day,inp_mon,inp_year,inp_week,inp_hol,out = conversion(week,days,months,years,list_train)
inp_day = np.array(inp_day)
inp_mon = np.array(inp_mon)
inp_year = np.array(inp_year)
inp_week = np.array(inp_week)
inp_hol = np.array(inp_hol)


# In[ ]:


plt.plot(result,color='red',label='predicted')
plt.plot(test_sales,color='purple',label="actual")
plt.xlabel("Date")
plt.ylabel("Sales")
leg = plt.legend()
plt.show()

