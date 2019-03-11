
İleti dizisi açıldı. 1 okunmuş ileti.

İçeriğe atla
Gmail ürününü ekran okuyucularla birlikte kullanma
in:sent 

İleti Dizileri
15 GB'lık kotanın 1,44 GB'ı (%9) kullanılıyor
Yönet
Şartlar · Gizlilik · Program Politikaları
Son hesap etkinliği: 1 dakika önce
Ayrıntılar

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 10)

np.random.seed = 42

df = pd.read_csv('data.csv')
df = df.drop('SampleNo', axis=1)

X = df.values[:100]
np.random.shuffle(X)
X_train = X[:, :-1]
y_train = X[:, -1]

label_lst = []
score_lst = []

# Linear Regression Model

from sklearn.linear_model import LinearRegression

label = 'Linear Regression'

mdl = LinearRegression()

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# Ridge Regression

from sklearn.linear_model import Ridge

label = 'Ridge'

mdl = Ridge(alpha=.5)

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# Lasso Regression

from sklearn.linear_model import Lasso

label = 'Lasso'

mdl = Lasso()

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# K Nearest Model

from sklearn.neighbors import KNeighborsRegressor

label = 'KNR (5)'

mdl = KNeighborsRegressor(n_neighbors=5)

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# Decision Tree

from sklearn.tree import DecisionTreeRegressor

label = 'DecisionTree'

mdl = DecisionTreeRegressor()

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# Random Forest

from sklearn.ensemble import RandomForestRegressor

label = 'RandomForest'

mdl = RandomForestRegressor(n_estimators=100)

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# Support Vector Regression

from sklearn.svm import SVR

label = 'SVR(rbf)'

mdl = SVR(kernel='rbf', gamma='scale')

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

label = 'SVR(linear)'

mdl = SVR(kernel='linear', gamma='scale')

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# Spline Regression

from pyearth import Earth as earth

label = 'Spline Regression'

mdl = earth(enable_pruning = True, penalty = 7, minspan_alpha = 1, endspan_alpha = 0.05)

r2_score_lst = cross_val_score(mdl, X_train, y_train, cv=5)
r2_score_mean = r2_score_lst.mean() * 100
print(label)
print(r2_score_mean)

label_lst.append(label)
score_lst.append(r2_score_mean)

# Best Model
mdl = RandomForestRegressor(n_estimators=100)
mdl.fit(X_train, y_train)
y_predict = mdl.predict(X_train)
training_r2_score = r2_score(y_train, y_predict) * 100
print('Best model training set r2_score')
print(training_r2_score)

# Predict unknown data using the best model
X_test = df.values[100:][:, :-1]
y_test = mdl.predict(X_test)

print('Predicted values for test set')
for prediction in y_test:
    print(prediction)

plt.bar(label_lst, score_lst)
plt.xlabel('Models')
plt.ylabel('Cross Validation r2 score')
plt.show()
regression.py
regression.py görüntüleniyor.
