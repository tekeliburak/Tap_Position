# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:37:29 2022

@author: previ
"""

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import numpy as np

deneme = pd.read_excel('case5_c2tap_load_pv_c1power.xlsx')
model = MultiOutputRegressor(LinearRegression())
temp = 8000
X_train = deneme.iloc[:temp, 6:].copy()
Y_train = deneme.iloc[:temp, :6].copy()
X_test = deneme.iloc[temp:, 6:]
Y_test = deneme.iloc[temp:,:6]


model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

round_off_values = np.round_(Y_pred)

df = pd.DataFrame(Y_pred, columns = ['Tap1_1','Tap1_2','Tap1_3','Tap2_1','Tap2_2','Tap2_3'])
df2 = pd.DataFrame(round_off_values, columns = ['Tap1_1','Tap1_2','Tap1_3','Tap2_1','Tap2_2','Tap2_3'])

temp1 = 2000
for i in range(temp1):
    if df2.Tap1_1[i] > 16:
        df2.Tap1_1[i] = 16
    if df2.Tap1_1[i] < -16:
        df2.Tap1_1[i] = -16
    if df2.Tap1_1[i] == -0:
        df2.Tap1_1[i] = 0

for i in range(temp1):
    if df2.Tap1_2[i] > 16:
        df2.Tap1_2[i] = 16
    if df2.Tap1_2[i] < -16:
        df2.Tap1_2[i] = -16
    if df2.Tap1_2[i] == -0:
        df2.Tap1_2[i] = 0

for i in range(temp1):
    if df2.Tap1_3[i] > 16:
        df2.Tap1_3[i] = 16
    if df2.Tap1_3[i] < -16:
        df2.Tap1_3[i] = -16
    if df2.Tap1_3[i] == -0:
        df2.Tap1_3[i] = 0
        
for i in range(temp1):
    if df2.Tap2_1[i] > 16:
        df2.Tap2_1[i] = 16
    if df2.Tap2_1[i] < -16:
        df2.Tap2_1[i] = -16
    if df2.Tap2_1[i] == -0:
        df2.Tap2_1[i] = 0
        
for i in range(temp1):
    if df2.Tap2_2[i] > 16:
        df2.Tap2_2[i] = 16
    if df2.Tap2_2[i] < -16:
        df2.Tap2_2[i] = -16
    if df2.Tap2_2[i] == -0:
        df2.Tap2_2[i] = 0
        
for i in range(temp1):
    if df2.Tap2_3[i] > 16:
        df2.Tap2_3[i] = 16
    if df2.Tap2_3[i] < -16:
        df2.Tap2_3[i] = -16
    if df2.Tap2_3[i] == -0:
        df2.Tap2_3[i] = 0

temp = 50        
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
ax[0][0].plot(range(temp),Y_test.c2_Tap1_1[0:temp], 'ro', label="Actual Value")
ax[0][0].plot(range(temp),df2.Tap1_1[0:temp], 'bo', label="Predict Value: Tap1_1")
ax[0][0].set_title('Tap1_1')

ax[0][1].plot(range(temp),Y_test.c2_Tap1_2[0:temp], 'ro', label="Actual Value")
ax[0][1].plot(range(temp),df2.Tap1_2[0:temp], 'go', label="Predict Value:Tap1_2")
ax[0][1].set_title('Tap1_2')

ax[0][2].plot(range(temp),Y_test.c2_Tap1_3[0:temp], 'ro', label="Actual Value")
ax[0][2].plot(range(temp),df2.Tap1_3[0:temp], 'co', label="Predict Value:Tap1_3")
ax[0][2].set_title('Tap1_3')

ax[1][0].plot(range(temp),Y_test.c2_Tap2_1[0:temp], 'ro', label="Actual Value")
ax[1][0].plot(range(temp),df2.Tap2_1[0:temp], 'mo', label="Predict Value:Tap2_1")
ax[1][0].set_title('Tap2_1')

ax[1][1].plot(range(temp),Y_test.c2_Tap2_2[0:temp], 'ro', label="Actual Value")
ax[1][1].plot(range(temp),df2.Tap2_2[0:temp], 'yo', label="Predict Value:Tap2_2")
ax[1][1].set_title('Tap2_2')

ax[1][2].plot(range(temp),Y_test.c2_Tap2_3[0:temp], 'ro', label="Actual Value")
ax[1][2].plot(range(temp),df2.Tap2_3[0:temp], 'ko', label="Predict Value:Tap2_3")
ax[1][2].set_title('Tap2_3')

fig, bx = plt.subplots(2, 3, sharex='col', sharey='row')
bx[0][0].plot(range(temp),Y_test.c2_Tap1_1[0:temp], 'r', label="Actual Value")
bx[0][0].plot(range(temp),df2.Tap1_1[0:temp], 'b', label="Predict Value: Tap1_1")
bx[0][0].set_title('Tap1_1')

bx[0][1].plot(range(temp),Y_test.c2_Tap1_2[0:temp], 'r', label="Actual Value")
bx[0][1].plot(range(temp),df2.Tap1_2[0:temp], 'g', label="Predict Value:Tap1_2")
bx[0][1].set_title('Tap1_2')

bx[0][2].plot(range(temp),Y_test.c2_Tap1_3[0:temp], 'r', label="Actual Value")
bx[0][2].plot(range(temp),df2.Tap1_3[0:temp], 'c', label="Predict Value:Tap1_3")
bx[0][2].set_title('Tap1_3')

bx[1][0].plot(range(temp),Y_test.c2_Tap2_1[0:temp], 'r', label="Actual Value")
bx[1][0].plot(range(temp),df2.Tap2_1[0:temp], 'm', label="Predict Value:Tap2_1")
bx[1][0].set_title('Tap2_1')

bx[1][1].plot(range(temp),Y_test.c2_Tap2_2[0:temp], 'r', label="Actual Value")
bx[1][1].plot(range(temp),df2.Tap2_2[0:temp], 'y', label="Predict Value:Tap2_2")
bx[1][1].set_title('Tap2_2')

bx[1][2].plot(range(temp),Y_test.c2_Tap2_3[0:temp], 'r', label="Actual Value")
bx[1][2].plot(range(temp),df2.Tap2_3[0:temp], 'k', label="Predict Value:Tap2_3")
bx[1][2].set_title('Tap2_3')

MSE1 = mean_squared_error(Y_test.c2_Tap1_1, df2.Tap1_1)
RMSE1 = math.sqrt(MSE1)
print('Tap1_1  MSE: ',MSE1)
print('Tap1_1 RMSE: ',RMSE1)

MSE2 = mean_squared_error(Y_test.c2_Tap1_2, df2.Tap1_2)
RMSE2 = math.sqrt(MSE2)
print('\nTap1_2  MSE: ',MSE2)
print('Tap1_2 RMSE: ',RMSE2)

MSE3 = mean_squared_error(Y_test.c2_Tap1_3, df2.Tap1_3)
RMSE3 = math.sqrt(MSE3)
print('\nTap1_3  MSE: ',MSE3)
print('Tap1_3 RMSE: ',RMSE3)

MSE4 = mean_squared_error(Y_test.c2_Tap2_1, df2.Tap2_1)
RMSE4 = math.sqrt(MSE4)
print('\nTap2_1  MSE: ',MSE4)
print('Tap2_1 RMSE: ',RMSE4)

MSE5 = mean_squared_error(Y_test.c2_Tap2_2, df2.Tap2_2)
RMSE5 = math.sqrt(MSE5)
print('\nTap2_2  MSE: ',MSE5)
print('Tap2_2 RMSE: ',RMSE5)

MSE6 = mean_squared_error(Y_test.c2_Tap2_3, df2.Tap2_3)
RMSE6 = math.sqrt(MSE6)
print('\nTap2_3  MSE: ',MSE6)
print('Tap2_3 RMSE: ',RMSE6)

prediction = df2.iloc[:50,:]
prediction.to_excel('c5_LR_pred.xlsx')