#!/usr/bin/env python
# coding: utf-8

# In[8]:


#importing libraries
import pandas as pd
import numpy as np 
from tqdm.auto import tqdm

#importing libraries of sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[2]:


#we are cleaning dataset here
df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ','_')

categorical_columns  = list(df.dtypes[df.dtypes=='object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')

df['totalcharges'] = pd.to_numeric(df['totalcharges'],errors='coerce')
df['totalcharges'] = df['totalcharges'].fillna(0)

df['churn'] = (df['churn']=='yes').astype(int)


# In[3]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[4]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[5]:


def train(df,y,C=1.0):
    dicts = df[categorical+numerical].to_dict(orient = 'records')

    dv = DictVectorizer(sparse = False)
    X = dv.fit_transform(dicts)

    model = LogisticRegression(C=C,max_iter=5000)
    model.fit(X,y)

    return dv, model


# In[6]:


def predict(df, dv, model):
    dicts = df[categorical+numerical].to_dict(orient = 'records')

    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:,1]

    return y_pred
    


# In[9]:


n_splits = 5
for C in tqdm([0.001,0.01,0.1,0.5,1,5,10]): # for different regularization
    print(f"for C = {C}")
    auc_scores = []
    kfold = KFold(n_splits=n_splits,shuffle=True,random_state=1)
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        
        y_train = df_train['churn'].values
        y_val = df_val['churn'].values
        
        dv,model = train(df_train,y_train,C=C)
        y_pred = predict(df_val,dv,model)
    
        auc = roc_auc_score(y_val,y_pred)
        auc_scores.append(auc)
    print(f"mean_auc: {np.mean(auc_scores):.3f} +- {np.std(auc_scores):.3f}\n")


# In[14]:


C=0.1
dv,model = train(df_full_train,df_full_train['churn'].values,C=C)
y_pred = predict(df_test,dv,model)

auc = roc_auc_score(y_test,y_pred)
auc


# ### save the model
# 
# - Save the model using pickle
# - 

# In[1]:


import pickle


# In[17]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[22]:


output_file = f'model_C={C}.bin'
output_file


# In[26]:


with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)


# ## load the model

# In[3]:


input_file = 'model_C=0.1.bin'
with open(input_file,'rb') as f_in:
    dv,model = pickle.load(f_in)


# In[4]:


dv,model


# In[18]:


X = dv.transform([customer])


# In[19]:


model.predict_proba(X)[0,1]


# In[ ]:




