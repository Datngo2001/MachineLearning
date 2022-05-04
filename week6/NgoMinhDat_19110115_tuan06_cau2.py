# -*- coding: utf-8 -*-
"""
Created on Sun May  1 09:00:49 2022

@author: ngomi
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib # Note: require sklearn v0.22+ (to update sklearn: pip install -U scikit-learn ). For old version sklearn: from sklearn.externals import joblib 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

train_set = pd.read_csv('NgoMinhDat_19110115_tuan06_data_train.csv')
test_set = pd.read_csv('NgoMinhDat_19110115_tuan06_data_test.csv')

train_set_labels = train_set["label"].copy()
train_set = train_set.drop(columns = "label") 
test_set_labels = test_set["label"].copy()
test_set = test_set.drop(columns = "label") 
    
X_train = train_set.to_numpy()
X_test = test_set.to_numpy()
y_train = train_set_labels.to_numpy()
y_test = test_set_labels.to_numpy()

def plot_(data, label = 'unspecified', showed=True):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("Name: " + str(label))
    #plt.axis("off")
    if showed:
        plt.show()
sample_id = 0
plot_(X_train[sample_id], y_train[sample_id])

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

grid_search = None
new_run = False
if new_run == True:
  param_grid = {'n_neighbors': [2, 3], 'weights': ['uniform', 'distance']}
  knei = KNeighborsClassifier()
  grid_search = GridSearchCV(knei, param_grid, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  joblib.dump(grid_search,'saved_var/knei_mul')
else:
  grid_search = joblib.load('saved_var/knei_mul')

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)

y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Thu lan 2

grid_search = None
new_run = False
if new_run == True:
  param_grid = {'n_neighbors': [3, 4, 5], 'weights': ['distance']}
  knei = KNeighborsClassifier()
  grid_search = GridSearchCV(knei, param_grid, n_jobs=4)
  grid_search.fit(X_train, y_train)
  joblib.dump(grid_search,'saved_var/knei_mul_2')
else:
  grid_search = joblib.load('saved_var/knei_mul_2')

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)

y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))


    




