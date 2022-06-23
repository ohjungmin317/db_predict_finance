# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:24:00 2022

@author: 오정민
"""
import pandas as pd
import numpy as np
import os
#import FinanceDataReader as fdr

from sklearn.linear_model import LinearRegression
# from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from numpy import sqrt
from numpy import mean_squared_error
import shap

# df = pd.read_csv("C:/Users/User/jungmin_finance/df.csv")
# df1 = pd.read_csv("C:/Users/User/jungmin_finance/df1.csv")
df3 = pd.read_csv("C:/Users/U ser/jungmin_finance/df3.csv")


df3.dtypes

df3.columns


df3_EDA = df3.drop(['DATE'], axis=1)
df3_EDA.dtypes
df3_tempo = df3.drop(['DATE'], axis=1)
df3_tempo.dtypes


def minmax_norm2(df_input):
    return 2*(df_input - df_input.min()) / ( df_input.max() - df_input.min())-1

df3_tempo = df3_tempo.drop(['KODEX 200 End_price'], axis=1) #change는 빼주기 >> 원래분포가 -1에서 1사이로 보임 굳이 변환할필요 없어보인다!
df3_tempo
df3_tempo.dtypes


df3_tempo=minmax_norm2(df3_tempo)

dummies = pd.get_dummies(df3_EDA[['KODEX 200 End_price']])

X = df3_tempo
y =dummies
#df1_EDA['Price']
y.dtypes

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) 
# train/test 비율을 7:3
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 확인

X

lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': None,
            'learning_rate': 0.01, # Step Size
            'n_estimators': 1000, # Number of trees
            'objective': 'regression'} # 목적 함수 (L2 Loss)

from sklearn.metrics import mean_squared_error
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
lgb_model_predict = lgb_model.predict(test_x) # test data 예측
print("RMSE: {}".format(sqrt(mean_squared_error(lgb_model_predict, test_y)))) # RMSE

explainer = shap.TreeExplainer(lgb_model) # Tree model Shap Value 확인 객체 지정
shap_values = explainer.shap_values(test_x) # Shap Values 계산

shap.initjs() # javascript 초기화 (graph 초기화)
shap.force_plot(explainer.expected_value, shap_values[1,:], test_x.iloc[1,:])
shap.force_plot(explainer.expected_value, shap_values, test_x) 
shap.summary_plot(shap_values, test_x)
