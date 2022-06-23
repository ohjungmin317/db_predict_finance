# -*- coding: utf-8 -*-
"""
Created on Sat May 21 21:11:42 2022

@author: 오정민
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
# from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
#matplotlib inline
from datetime import datetime



Jungmin = pd.read_csv("jungmin_Finance.csv")

fname_input = './jungmin_Finance.csv'
data = pd.read_csv(fname_input)

pd.options.display.max_rows=100
pd.options.display.max_columns=100


print(data.info())
print(data.describe())


    

plt.figure(figsize=(20,20)) 
sns.set(font_scale=1.5)
sns.heatmap(Jungmin.corr(),
               annot = True, #실제 값 화면에 나타내기
           cmap = 'Greens', #색상
          )

  
