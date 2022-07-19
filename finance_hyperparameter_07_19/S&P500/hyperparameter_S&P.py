# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:11:38 2022

@author: 오정민
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def minmax_norm2(df_input):
    return 2*(df_input - df_input.min()) / ( df_input.max() - df_input.min())-1

fname_input = './S&P500.csv'
data = pd.read_csv(fname_input)


print(data.info())

data = data.iloc[:, 1:]
print(data.info())

X = data[['AD', 'STO', 'Q-stick', 'ETF', 'KOSDAQ','Dollar', 'STOXX', 'CSI']]

X.dtypes
X

dummies = pd.get_dummies(data[['Price']])
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


# 데이터의 스케일 정보를 확인
# - 전처리 여부를 판단
import pandas as pd
X_df = pd.DataFrame(X)
# 스케일의 편차가 존재하기 때문에
# 정규화 처리가 필요함
print(X_df.describe(include='all'))

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler().fit(X1_train)
X1_train = scaler.transform(X1_train)
X1_test = scaler.transform(X1_test)


param_grid = {'max_depth':[1, 2, 3, 4, 5, 6, None],
              'min_samples_split':[1, 2, 3, 4, 5, 6],
              'criterion':['gini', 'entropy','log_loss'],
              'splitter':['best','random'],
              #'min_samples_leaf':[1, 2, 3, 4, 5, 6],
              #'min_weight_fraction_leaf':[0.01, 0, 0.1, 0.5],
              #'random_state':[1,2,3,4,5,100]
              'class_weight':['balanced',{2:5},{1:1},{3:1},{4:1}],
              'max_features':[0.1, 0.01, 0.5]}

# 다수개의 딕셔너리를 제공함 -> 각 제약에 맞는 값을 지정해주면 에러 값이 따로 뜨지 않는다. 


from sklearn.model_selection import GridSearchCV
cv=KFold(n_splits=5,shuffle=True,random_state=1)
estimator=DecisionTreeClassifier()

grid_model = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=-1).fit(X1_train,y_train)

# 모든 하이퍼 파라메터를 조합하여 평가한 
# 가장 높은 교차검증 SCORE 값을 반환
print(f'best_score -> {grid_model.best_score_}')
# 가장 높은 교차검증 SCORE 가 어떤 
# 하이퍼 파라메터를 조합했을 때 만들어 졌는지 확인
print(f'best_params -> {grid_model.best_params_}')
# 가장 높은 교차검증 SCORE의 
# 하이퍼 파라메터를 사용하여 생성된 모델 객체를 반환
print(f'best_model -> {grid_model.best_estimator_}')

score = grid_model.score(X1_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = grid_model.score(X1_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')