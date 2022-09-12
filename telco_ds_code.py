#!/usr/bin/env python
# coding: utf-8


# In[2]:


import warnings
warnings.filterwarnings("ignore")

get_ipython().system('pip install imblearn')


# In[3]:


##libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import warnings
import scipy
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report




# In[4]:


#Preparation and cleaning of the data
df = pd.read_csv('telco.csv', sep=',')#reading dataset


# In[5]:


df.info()


# In[6]:


df.columns = df.columns.str.replace(' ', '')  #remove empty characters from the columns


# In[7]:


print(df.shape)


# In[8]:


df.head()


# In[9]:


print(df.duplicated().value_counts())  #get duplicated rows


# In[10]:


#Data Quality Report for categorical features
categorical_columns= df.select_dtypes(exclude=["number"]).columns
formatter = "{0:.2f}"
listModeName=[]
list_mode=[]
print(f"{'Feature' : <16}{'Count' : ^10}{'Card' : ^10}{'Mode':^16}{'Mode Freq':^15}{'Mode(%)':^15}{'2nd Mode':^15}{'2nd Mode Freq':^15}{'2nd Mode(%)':^15}")
for i in range(len(categorical_columns)):
    listModeName.extend(df[categorical_columns[i]].value_counts().index.tolist())  
    list_mode.extend(df[categorical_columns[i]].value_counts())  
    print(f"{categorical_columns[i]: <16}{df[categorical_columns[i]].count(): ^10}{df[categorical_columns[i]].nunique(): ^10}{df[categorical_columns[i]].mode().values[0]: ^16}{df[categorical_columns[i]].value_counts().max(): ^15}{formatter.format((df[categorical_columns[i]].value_counts().max()/df[categorical_columns[i]].count())*100): ^15}{listModeName[1]: ^15}{list_mode[1]: ^15}{formatter.format((list_mode[1]/df[categorical_columns[i]].count())*100): ^15}")
    list_mode.clear() 
    listModeName.clear()


# In[11]:


TotalCharges_median = df[df['TotalCharges'] != ' ']['TotalCharges'].apply(pd.to_numeric).median() #get the median without the empty strings
df["TotalCharges"].replace([' '], TotalCharges_median,inplace = True) #fil empty values with median
df['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric) #make the column type numeric 
print(TotalCharges_median, '\n')


# In[12]:


#Data Quality Report for numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
dqr_num = df[numeric_cols].describe()
cardinality = df.apply(pd.Series.nunique)
dqr_num.loc['cardinality'] = cardinality[dqr_num.columns]
dqr_num = dqr_num.T

print(dqr_num)


# In[13]:


df['Churn'].value_counts()


# In[14]:


df['TotalCharges'].sort_values(ascending=False)


# In[15]:


df.drop(columns = ["customerID"],inplace = True)  #remove the customerID column


# In[16]:


df.info()


# In[17]:


#encoding the features that have 2 values
binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'PhoneService', 'Churn'] 

for i in binary_cols: 
    if(i == 'gender'):
        df[i].replace({'Female': 0, 'Male': 1}, inplace=True) 
    else:
        df[i].replace({'No': 0, 'Yes': 1}, inplace=True) 


# In[18]:


#encoding the features that have more than 2 values
encode_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod", "OnlineBackup"]
ohe = OneHotEncoder(sparse = False)

for i in encode_cols:
    df = pd.concat((df , pd.DataFrame(ohe.fit_transform(df[i].to_frame()),columns = str(i+"_") + np.sort(df[i].unique()))),axis = 1)
    df.drop(columns = [i],inplace = True)
    


# In[19]:


print(df.columns)


# In[20]:


scaler = MinMaxScaler(feature_range=(0, 1), copy=True)  #max min normalization [0,1]
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
#df


# In[21]:


df['tenure'].value_counts()


# In[22]:


df.columns = df.columns.str.replace(' ', '')  #remove empty characters on the columns which occured after the encoding
df


# In[23]:


df.info()  #check the types and names of the columns
# Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = df.corr().abs()

# Getting the upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
print(list(to_drop))
df = df.drop(columns = to_drop)
df.shape


# In[24]:


plt.figure(figsize=(15,8))
df.corr()['Churn'].sort_values(ascending = False).plot(kind='bar') #Correlation of "Churn" with other features:


# In[25]:


# For Test and Train - Normal, %90, %10
y = df.Churn.to_frame()
X = df.drop(columns = ["Churn"])
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.25, random_state = 10)


# In[26]:


# Models
# Train and Test
models = [LogisticRegression(),
          GaussianNB(),
          SGDClassifier(),
          KNeighborsClassifier(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          SVC(),
          AdaBoostClassifier(), 
          XGBClassifier(),
          MLPClassifier()
         ]

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    print(models[i], ':', model.score(X_test, y_test))
    y_pred = model.predict(X_test)

#Confusion Matrix

    plt.figure(figsize=(14,12))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,fmt = 'd',linecolor="k",linewidths=3)

    plt.title("CONFUSION MATRIX",fontsize=20)
    plt.show()

# Evaulation Metrics 
    
    report = classification_report(y_test, y_pred)
    print(report)
    
    from sklearn.metrics import r2_score
    r2score = r2_score(y_test, y_pred)
    print('R2 Score: ', r2score)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error: ', mae)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error' ,mse)
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    sns.set()

    plt.plot(fpr, tpr)

    plt.plot(fpr, fpr, linestyle = '--', color = 'k')

    plt.xlabel('False positive rate')

    plt.ylabel('True positive rate')

    AUROC = np.round(roc_auc_score(y_test, y_pred), 2)

    plt.title(f'ROC curve; AUROC: {AUROC}');

    plt.show()


# In[27]:


# SMOTE - Oversampling
# Oversampled Test and Train, %90, %10
sm = SMOTE()
X_sm , y_sm = sm.fit_resample(X, y)
y_sm.Churn.value_counts()


# In[28]:


#Test and train oversampled
get_ipython().system('pip install sklearn')
X_train_sm , X_test_sm , y_train_sm , y_test_sm = train_test_split(X_sm, y_sm, test_size = 0.25, random_state = 10)
for i, model in enumerate(models):
    model.fit(X_train_sm, y_train_sm)
    print(models[i], ':', model.score(X_test_sm, y_test_sm))
    y_pred_sm = model.predict(X_test_sm)
    plt.figure(figsize=(14,12))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(y_test_sm, y_pred_sm),
            annot=True,fmt = 'd',linecolor="k",linewidths=3)

    plt.title("CONFUSION MATRIX",fontsize=20)
    plt.show()
 

    report = classification_report(y_test_sm, y_pred_sm)
    print(report)
    
    from sklearn.metrics import r2_score
    r2score = r2_score(y_test_sm, y_pred_sm)
    print('R2 Score: ', r2score)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_sm, y_pred_sm)
    print('Mean Absolute Error: ', mae)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test_sm, y_pred_sm)
    print('Mean Squared Error' ,mse)
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds = roc_curve(y_test_sm, y_pred_sm)
    sns.set()

    plt.plot(fpr, tpr)

    plt.plot(fpr, fpr, linestyle = '--', color = 'k')

    plt.xlabel('False positive rate')

    plt.ylabel('True positive rate')

    AUROC = np.round(roc_auc_score(y_test_sm, y_pred_sm), 2)

    plt.title(f'ROC curve; AUROC: {AUROC}');

    plt.show()



# In[29]:


# Oversampled, Smoted, Cross Validation

X_train_sm , X_test_sm , y_train_sm , y_test_sm = train_test_split(X_sm, y_sm, test_size = 0.25, random_state = 10)
for i, model in enumerate(models):
    model.fit(X_train_sm, y_train_sm)
    print(models[i], ':', model.score(X_test_sm, y_test_sm))
    y_pred_sm = model.predict(X_test_sm)


# In[30]:



 
#Implementing cross validation
k = 10
kf = KFold(n_splits=k, random_state=None)



for i, model in enumerate(models):


    sum_conf_matrixes = np.zeros((2, 2))
    
    divided_by = 0
    zerozero, onezero, zeroone, oneone = 0, 0, 0, 0
    
    
    acc_score = []
    for train_index , test_index in kf.split(X_sm):
        X_train , X_test = X_sm.iloc[train_index,:],X_sm.iloc[test_index,:]
        y_train , y_test = y_sm.iloc[train_index,:],y_sm.iloc[test_index,:]

        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)

        acc = accuracy_score(pred_values , y_test)
        acc_score.append(acc)
        
        sum_conf_matrixes += confusion_matrix(y_test, pred_values)
        divided_by = divided_by + 1


    avg_acc_score = sum(acc_score)/k
    sum_conf_matrixes = sum_conf_matrixes / divided_by


    print(models[i], ' Avg accuracy : {}'.format(avg_acc_score))

    
    plt.figure(figsize=(14,12))
    plt.subplot(221)
    sns.heatmap(sum_conf_matrixes,
            annot=True,fmt = 'g',linecolor="k",linewidths=3)

    plt.title("CONFUSION MATRIX",fontsize=20)
    plt.show()

    


# In[ ]:




