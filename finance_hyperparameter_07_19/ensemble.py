# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:24:54 2022

@author: User
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# import openpyxl
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import os
# import FinanceDataReader as fdr

from sklearn.linear_model import LinearRegression
from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from numpy import sqrt
from sklearn.metrics import mean_squared_error
import shap

def minmax_norm2(df_input):
    return 2*(df_input - df_input.min()) / ( df_input.max() - df_input.min())-1


df5 = pd.read_csv("C:/Users/User/jungmin_finance/Dollar.csv")
df5.columns

#Dollar X = df5[['bull', 'AD', 'Force', 'etf', 'Williams', 'nav','volume', 'D']]

#WTI X = df5[['WTI Enhanced(H) Volume', 'AD', 'STO', 'Force','Q-stick', 'Q-stick.1', 'CCI', 'Company', 'Foreign', 'Person', 'NAV','ETF', 'Bull']]

# Gold X = df5[['AD', 'STO', 'Force', 'Q-stick', 'Q-stick.1', 'CCI','Company', 'NAV', 'ETF', 'Bull', 'Bear']]
       
       

# Nikkei X = df5[['KOSDAQ', 'EURO', 'WTI', 'Dollar', 'AD', 'STO', 'CCI','NAV', 'ETF']]
    
# CSI X = df5[['Force', 'Q-stick', 'NAV', 'ETF', 'AD','Q-stick.1', 'STO', 'KODEX', 'KOSDAQ', 'WTI', 'Dollar']]

# STOXX X = df5[['Q-stick', 'NAV', 'ETF', 'Bull', 'Gold', 'WTI','Dollar', 'KODEX']]

# S&P X = df5[['AD', 'STO', 'Q-stick', 'ETF', 'KOSDAQ','Dollar', 'STOXX', 'CSI']]

# KOSDAQ X = df5[['Beta', ' KOSDAQ150 Volume', 'CCI', 'NAV', 'Q-stick','Gold', 'WTI', 'KODEX', 'STOXX']]

#KODEX X = df5[['Force', 'ETF', 'Qstick', 'Bull', 'STO', 'CCI','Williams', 'company', 'Dollar', 'Kosdaq', 'WTI', 'Gold']]
    
X.dtypes
X

dummies = pd.get_dummies(df5[['Price']])
y = dummies[6:1528]
y

X2 = minmax_norm2(X)
X2.dtypes

X1 = X2[6:1528]
X1
result1 = X2.iloc[1528:1529]
result1

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=1)
X1_train, X1_val, y_train, y_val = train_test_split(X1_train, y_train, test_size=0.25, random_state=1)

#pca = PCA(n_components=3) # 주성분을 몇개로 할지 결정
#printcipalComponents = pca.fit_transform(X1)
#X1 = pd.DataFrame(data=printcipalComponents, columns = ['p1','p2','p3'])

#sum(pca.explained_variance_ratio_)



xgb = XGBClassifier(booster='gbtree',
                    colsample_bylevel=0.9,
                    colsample_bytree=0.8,
                    gamma=1,
                    max_depth=None,
                    min_child_weight=3,
                    n_estimators=50,
                    nthread=4,
                    objective='binary:logistic',
                    random_state=2,
                    silent=True,
                    use_label_encoder=False,  eval_metric='logloss')

lgbm = LGBMClassifier(
    max_depth=14,
    min_child_weight=3, n_estimators=400)

decisiontree = DecisionTreeClassifier(max_depth=None)

randomforest = RandomForestClassifier(random_state=156,n_jobs=-1)

adaboost = AdaBoostClassifier()

lr = LogisticRegression()

knn = KNeighborsClassifier()

model = [xgb, lgbm, decisiontree, randomforest, adaboost, lr, knn]

model1 = xgb.fit(X1_train, y_train.values.ravel(), eval_metric="rmse")
pred_XGB = model1.predict(X1_val)
xgb_acc = accuracy_score(y_val, pred_XGB)
print(f'XGB 정확도: {xgb_acc:.4f}')
f1 = f1_score(y_val, pred_XGB)
print(f'XGB f1 score: {f1:.4f}')

model2 = lgbm.fit(X1_train, y_train.values.ravel())
pred_LGBM = model2.predict(X1_val)
lgbm_acc = accuracy_score(y_val, pred_LGBM)
print(f'LGBM 정확도: {lgbm_acc:.4f}')
f1 = f1_score(y_val, pred_LGBM)
print(f'LGBMGB f1 score: {f1:.4f}')

model3 = decisiontree.fit(X1_train, y_train.values.ravel())
pred_dt = model3.predict(X1_val)
dt_acc = accuracy_score(y_val, pred_dt)
print(f'Decision Tree 정확도: {dt_acc:.4f}')
f1 = f1_score(y_val, pred_dt)
print(f'Decision Tree f1 score: {f1:.4f}')

model4 = randomforest.fit(X1_train, y_train.values.ravel())
pred_rf = model4.predict(X1_val)
rf_acc = accuracy_score(y_val, pred_rf)
print(f'Random Forest 정확도: {rf_acc:.4f}')
f1 = f1_score(y_val, pred_rf)
print(f'RandomForest f1 score: {f1:.4f}')

model5 = adaboost.fit(X1_train, y_train.values.ravel())
pred_ab = model5.predict(X1_val)
ab_acc = accuracy_score(y_val, pred_ab)
print(f'Adaboost 정확도: {ab_acc:.4f}')
f1 = f1_score(y_val, pred_ab)
print(f'Adaaboost f1 score: {f1:.4f}')

model6 = lr.fit(X1_train, y_train.values.ravel())
pred_LR = model6.predict(X1_val)
lr_acc = accuracy_score(y_val, pred_LR)
print(f'Logistic Regression 정확도: {lr_acc:.4f}')
f1 = f1_score(y_val, pred_LR)
print(f'Logistic Regression f1 score: {f1:.4f}')

model7 = knn.fit(X1_train, y_train.values.ravel())
pred_KNN = model7.predict(X1_val)
knn_acc = accuracy_score(y_val, pred_KNN)
print(f'KNN 정확도: {knn_acc:.4f}')
f1 = f1_score(y_val, pred_KNN)
print(f'KNN f1 score: {f1:.4f}')

print('Test set')
model1 = xgb.fit(X1_train, y_train.values.ravel(), eval_metric="rmse")
pred_XGB = model1.predict(X1_test)
xgb_acc = accuracy_score(y_test, pred_XGB)
print(f'XGB 정확도: {xgb_acc:.4f}')
f1 = f1_score(y_test, pred_XGB)
print(f'XGB f1 score: {f1:.4f}')

model2 = lgbm.fit(X1_train, y_train.values.ravel())
pred_LGBM = model2.predict(X1_test)
lgbm_acc = accuracy_score(y_test, pred_LGBM)
print(f'LGBM 정확도: {lgbm_acc:.4f}')
f1 = f1_score(y_test, pred_LGBM)
print(f'LGBMGB f1 score: {f1:.4f}')

model3 = decisiontree.fit(X1_train, y_train.values.ravel())
pred_dt = model3.predict(X1_test)
dt_acc = accuracy_score(y_test, pred_dt)
print(f'Decision Tree 정확도: {dt_acc:.4f}')
f1 = f1_score(y_test, pred_dt)
print(f'Decision Tree f1 score: {f1:.4f}')

model4 = randomforest.fit(X1_train, y_train.values.ravel())
pred_rf = model4.predict(X1_test)
rf_acc = accuracy_score(y_test, pred_rf)
print(f'Random Forest 정확도: {rf_acc:.4f}')
f1 = f1_score(y_test, pred_rf)
print(f'RandomForest f1 score: {f1:.4f}')

model5 = adaboost.fit(X1_train, y_train.values.ravel())
pred_ab = model5.predict(X1_test)
ab_acc = accuracy_score(y_test, pred_ab)
print(f'Adaboost 정확도: {ab_acc:.4f}')
f1 = f1_score(y_test, pred_ab)
print(f'Adaaboost f1 score: {f1:.4f}')


# 교차검증

X1_train_new, X1_test_new, y_train_new, y_test_new = train_test_split(X1, y, test_size=0.2, random_state=1) 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score , cross_validate
# from sklearn.datasets import load_iris

Classifiers = [xgb, lgbm, decisiontree, randomforest, adaboost, lr, knn]
Classifiers_name=["XGB","LGBM","Decision Tree","Random Forest","Adaboost","Logistic Regression","KNN"]

for i in range(len(Classifiers)): 
    scores = cross_val_score(Classifiers[i] , X1_train_new, y_train_new.values.ravel(), scoring='accuracy',cv=3)
    print(Classifiers_name[i],'교차 검증별 정확도:',np.round(scores, 4))
    print(Classifiers_name[i],'평균 검증 정확도:', np.round(np.mean(scores), 4))
    
train_stack = np.array([pred_XGB,pred_LGBM,pred_LR,pred_rf,pred_ab,pred_KNN],dtype=object)
train_stack = np.transpose(train_stack)

train_stack

Total =decisiontree.fit(train_stack,y_test.values.ravel())
Total_pred = Total.predict(train_stack)
Total_acc = accuracy_score(y_test,Total_pred)
print(f'스태킹 모델 정확도: {Total_acc:.4f}')
f1 = f1_score(y_test, Total_pred)
print(f'스태킹 모델 f1 score: {f1:.4f}')

scores = cross_val_score(Total, train_stack, y_test.values.ravel(), scoring='accuracy',cv=3)
print('스태킹 교차 검증별 정확도:',np.round(scores, 3))
print('스태킹 평균 검증 정확도:', np.round(np.mean(scores), 4))

#찐예측

model=[model1,model2,model4,model5,model6,model7]

model_total_stacking = []


for i in range(6):
    model_total_stacking.append(model[i].predict(result1))
    print(model_total_stacking[i])

Total_pred1= Total.predict(np.transpose(model_total_stacking))
Total_pred1
    