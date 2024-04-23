#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter 
#from catboost import CatBoostClassifier


import warnings
warnings.filterwarnings("ignore")


# In[3]:


data = pd.read_csv('application_data.csv')
data


# In[4]:


data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].fillna('No Value')


# In[5]:


data['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE'].fillna(data['NAME_TYPE_SUITE'].mode())


# In[6]:


list_contact = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
list_region =  ['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION']
list_city = ['REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']
list_docs = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
list_rate = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']
list_other = ['TARGET', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED','DAYS_ID_PUBLISH', 'HOUR_APPR_PROCESS_START']


# In[7]:


print(len(list_contact), len(list_region), len(list_city), len(list_docs), len(list_rate), len(list_other))


# In[190]:


for i in list_contact:  
    plt.savefig('contacts.jpg')
    plt.subplots(figsize=(10, 3))
    plt.xticks(rotation = 'horizontal')
    plt.title(f'{i}')
    sns.countplot(x=i, hue= 'TARGET', data = data)


# In[9]:


corr = data[list_contact].corr()
fig1, ax1 = plt.subplots(figsize=(5, 5))
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            ax = ax1,annot=True)


# In[10]:


for i in list_region:    
    plt.subplots(figsize=(10, 5))
    plt.xticks(rotation = 'horizontal')
    plt.title(f'{i}')
    sns.countplot(x=i, hue= 'TARGET', data = data)


# In[11]:


corr = data[list_region].corr()
fig1, ax1 = plt.subplots(figsize=(3, 3))
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            ax = ax1,annot=True)


# In[12]:


for i in list_city:    
    plt.subplots(figsize=(10, 5))
    plt.xticks(rotation = 'horizontal')
    plt.title(f'{i}')
    sns.countplot(x=i, hue= 'TARGET', data = data)


# In[13]:


corr = data[list_city].corr()
fig1, ax1 = plt.subplots(figsize=(3, 3))
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            ax = ax1,annot=True)


# In[106]:


for i in list_docs:    
    plt.subplots(figsize=(10, 5))
    plt.xticks(rotation = 'horizontal')
    plt.title(f'{i}')
    sns.countplot(x=i, hue= 'TARGET', data = data)


# In[14]:


corr = data[list_docs].corr()
fig1, ax1 = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            ax = ax1,annot=True)


# In[15]:


data['docs'] = data['FLAG_DOCUMENT_2']+data['FLAG_DOCUMENT_3']+data['FLAG_DOCUMENT_4'] + data['FLAG_DOCUMENT_5'] + data['FLAG_DOCUMENT_6'] + data['FLAG_DOCUMENT_7'] + data['FLAG_DOCUMENT_8'] + data['FLAG_DOCUMENT_9'] + data['FLAG_DOCUMENT_10'] + data['FLAG_DOCUMENT_11'] + data['FLAG_DOCUMENT_12'] + data['FLAG_DOCUMENT_13'] + data['FLAG_DOCUMENT_14'] + data['FLAG_DOCUMENT_15'] + data['FLAG_DOCUMENT_16'] + data['FLAG_DOCUMENT_17'] + data['FLAG_DOCUMENT_18'] + data['FLAG_DOCUMENT_19'] +data['FLAG_DOCUMENT_20'] + data['FLAG_DOCUMENT_21']


# In[16]:


for i in list_rate:    
    plt.subplots(figsize=(10, 5))
    plt.xticks(rotation = 'horizontal')
    plt.title(f'{i}')
    sns.countplot(x=i, hue= 'TARGET', data = data)


# In[198]:


corr = data[list_rate].corr()
fig1, ax1 = plt.subplots(figsize=(3, 3))
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            ax = ax1,annot=True)


# In[17]:


plt.subplots(figsize=(10, 5))
plt.xticks(rotation = 'horizontal')
plt.title('HOUR_APPR_PROCESS_START')
sns.countplot(x='HOUR_APPR_PROCESS_START', hue= 'TARGET', data = data)


# In[18]:


unwanted_int = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'SK_ID_CURR','REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY', 'REGION_RATING_CLIENT_W_CITY']


# In[19]:


unwanted_obj = ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
df_obj = data.select_dtypes('object')
df_obj = df_obj.drop(unwanted_obj, axis = 1)
df_obj.describe().T


# In[20]:


df_int = data.select_dtypes('int')
df_int = df_int.drop(unwanted_int, axis = 1)
df_int.describe().T


# In[21]:


unwanted_float=['AMT_GOODS_PRICE', 
        'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1',
       'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
       'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
       'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
       'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
       'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
       'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
       'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
       'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
       'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
       'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
       'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
       'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
       'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
       'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE',  'OBS_60_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE']
df_float = data.select_dtypes('float')
df_float = df_float.drop(unwanted_float, axis = 1)
df_float.describe().T


# In[22]:


df_new = pd.concat([df_obj, df_int, df_float], axis = 1)


# In[23]:


df_new.columns
data=df_new
data.columns


# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['NAME_CONTRACT_TYPE'] = le.fit_transform(data['NAME_CONTRACT_TYPE'])
data['CODE_GENDER'] = le.fit_transform(data['CODE_GENDER'])
data['FLAG_OWN_CAR'] = le.fit_transform(data['FLAG_OWN_CAR'])
data['FLAG_OWN_REALTY'] = le.fit_transform(data['FLAG_OWN_REALTY'])
data['NAME_TYPE_SUITE'] = le.fit_transform(data['NAME_TYPE_SUITE'].astype(str))
data['NAME_INCOME_TYPE'] = le.fit_transform(data['NAME_INCOME_TYPE'])
data['NAME_EDUCATION_TYPE'] = le.fit_transform(data['NAME_EDUCATION_TYPE'])
data['NAME_FAMILY_STATUS'] = le.fit_transform(data['NAME_FAMILY_STATUS'])
data['NAME_HOUSING_TYPE'] = le.fit_transform(data['NAME_HOUSING_TYPE'])
data['OCCUPATION_TYPE'] = le.fit_transform(data['OCCUPATION_TYPE'].astype(str))
data['WEEKDAY_APPR_PROCESS_START'] = le.fit_transform(data['WEEKDAY_APPR_PROCESS_START'])
data['ORGANIZATION_TYPE'] = le.fit_transform(data['ORGANIZATION_TYPE'])


# In[ ]:





# In[25]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import sklearn.metrics as metrics
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, fbeta_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
import math


# In[26]:


from sklearn.svm import LinearSVC, SVC


# In[27]:


data.info()
data


# In[34]:


X = data.drop(['TARGET'],axis = 1)
y = data['TARGET']
X['COMMONAREA_AVG'] = X['COMMONAREA_AVG'].fillna(data['COMMONAREA_AVG'].mean())
X['AMT_REQ_CREDIT_BUREAU_HOUR'] = X['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(data['AMT_REQ_CREDIT_BUREAU_HOUR'].median())
X['AMT_REQ_CREDIT_BUREAU_DAY'] = X['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(data['AMT_REQ_CREDIT_BUREAU_DAY'].median())
X['AMT_REQ_CREDIT_BUREAU_WEEK'] = X['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(data['AMT_REQ_CREDIT_BUREAU_WEEK'].median())
X['AMT_REQ_CREDIT_BUREAU_MON'] = X['AMT_REQ_CREDIT_BUREAU_MON'].fillna(data['AMT_REQ_CREDIT_BUREAU_MON'].median())
X['AMT_REQ_CREDIT_BUREAU_QRT'] = X['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(data['AMT_REQ_CREDIT_BUREAU_QRT'].median())
X['AMT_REQ_CREDIT_BUREAU_YEAR'] = X['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(data['AMT_REQ_CREDIT_BUREAU_YEAR'].median())
X['OBS_30_CNT_SOCIAL_CIRCLE'] = X['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(data['OBS_30_CNT_SOCIAL_CIRCLE'].median())
X['DEF_30_CNT_SOCIAL_CIRCLE'] = X['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(data['DEF_30_CNT_SOCIAL_CIRCLE'].median())
X['DAYS_LAST_PHONE_CHANGE'] = X['DAYS_LAST_PHONE_CHANGE'].fillna(data['DAYS_LAST_PHONE_CHANGE'].mean())
X['EXT_SOURCE_2'] = X['EXT_SOURCE_2'].fillna(data['EXT_SOURCE_2'].mean())
X['EXT_SOURCE_3'] = X['EXT_SOURCE_3'].fillna(data['EXT_SOURCE_3'].mean())
X['AMT_ANNUITY'] = X['AMT_ANNUITY'].fillna(data['AMT_ANNUITY'].mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 2020)

    
scaler = RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
def model_Evaluate(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 2020)
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # accuracy of model on training data
    acc_train = model.score(X_train_scaled, y_train)
    # accuracy of model on test data
    acc_test = model.score(X_test_scaled, y_test)
    
    print('Accuracy of model on training data : {}'.format(acc_train*100))
    print('Accuracy of model on testing data : {} \n'.format(acc_test*100))

    # Predict values for Test dataset
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[::,1]
    
    # y_score
    try:
        y_score = model.fit(X_train_scaled, y_train).decision_function(X_test_scaled)
    except:
        pass
    
    try:
        clf = model.fit(X_train_scaled, y_train)
        y_score = clf.predict_proba(X_test_scaled)
    except:
        pass
    
    # precision of model on test data
    pre_test = precision_score(y_test, y_pred)
    
    # recall of model on test data
    rec_test = recall_score(y_test, y_pred)
    
    # f1 of model on test data
    f1_test = f1_score(y_test, y_pred)
    
    # f2 of model on test data
    f2_test = fbeta_score(y_test, y_pred, beta=2, average='macro')
    
    # AUC of model on test data
    auc_test = roc_auc_score(y_test, y_pred_proba)
    
    # Mattews_corrcoef
    mcc = matthews_corrcoef(y_pred, y_test)
    
    
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    print(f'f2 score: {f2_test}')
    print(f'matthews_corrcoef: {mcc}')
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Reds',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    
    d = {'Train_Accuracy': [acc_train], 'Test_Accuracy': [acc_test], 
         'Precision': [pre_test], 'Recall': [rec_test],
         'AUC': [auc_test], 'F1_Score': [f1_test], 'F2_Score': [f2_test], 
         'Roc_Auc_score': auc_test, 'Matthews_corrcoef' : mcc}
    
    df = pd.DataFrame(data=d)
    # summarize feature importance
    print(type(model).__name__)
    modelName = type(model).__name__

    # roc curve
    try:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    except:
        pass
    return df



# In[35]:


from sklearn.calibration import CalibratedClassifierCV
svm = LinearSVC()
clf = CalibratedClassifierCV(svm) 
clf.fit(X_train_scaled, y_train)
svm_df = model_Evaluate(clf, X, y)


# In[40]:


df=pd.read_csv('username.csv')


# In[74]:


USERNAME = df["USERNAME"]
PASSWORD = df["PASSWORD"]  

credentials = dict(zip(USERNAME, PASSWORD))


USERNAME = input("Enter username: ")
PASSWORD = input("Enter password: ")

# Check for matching credentials
if USERNAME in credentials and credentials[USERNAME] == PASSWORD:
    list1=[]
    for i in X.columns:
        d=float(input(i))
        list1.append(d)
    arr1=np.array([list1])
    list2=[]
    arr2=np.array([list2])
    arr2=clf.predict(arr1)
    if(arr2[0]==0):
        print('Fraudulent transaction')
    else:
        print('Please continue')
    print(arr2)
        
else:
   print("Invalid Username or Password")


# In[ ]:




