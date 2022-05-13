'''
The following code is mainly from Chap 4, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb

LAST REVIEW: March 2022
'''

# In[0]: IMPORTS, SETTINGS
#region
import sys
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import numpy as np
np.random.seed(42) # to output the same result across runs
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)       
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd") # Ignore useless warnings (see SciPy issue #5998)
font_size = 14
let_plot = True
#endregion

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
iris = datasets.load_iris()
from copy import deepcopy

print(iris.keys()) # data: features, target: label
#print(iris.DESCR) # description of the data

X = iris["data"][:, (2, 3)]   # petal length, petal width
y = iris["target"]  # use all 3 classes
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=15, shuffle=True)

softmax_reg = LogisticRegression(multi_class="multinomial", # multinomial: use Softmax regression
                                 solver="lbfgs", random_state=42,warm_start=True) # C=10

n_iter_wait = 200
minimum_val_error = np.inf  
train_errors, val_errors = [], []

from sklearn.metrics import mean_squared_error
for iter in range(1000):
    # Train and compute val. error:
    softmax_reg.fit(X_train, y_train)
    y_val_predict = softmax_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_val_predict)
    # Save the best model:
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_iter = iter
        best_model = deepcopy(softmax_reg)   
    # Stop after n_iter_wait loops with no better val. error:
    if best_iter + n_iter_wait < iter :
        break

    # Save for plotting purpose:
    val_errors.append(val_error)
    y_train_predict = softmax_reg.predict(X_val)
    train_errors.append(mean_squared_error(y_val, y_train_predict)) 
    
train_errors = np.sqrt(train_errors) # convert to RMSE
val_errors = np.sqrt(val_errors)
# Print best iter and model
print("best_iter:",best_iter)
best_model.intercept_, best_model.coef_

# Plot learning curves
if let_plot:
    best_val_error = val_errors[best_iter]
    plt.plot(val_errors, "b-", linewidth=2, label="Validation set")
    plt.plot(train_errors, "r-", linewidth=2, label="Training set")
    plt.annotate('Best model',xytext=(best_iter, best_val_error+0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 xy=(best_iter, best_val_error), ha="center", fontsize=font_size,  )      
    plt.xlim(0, iter)
    plt.grid()
    plt.legend(loc="upper right", fontsize=font_size)
    plt.xlabel("Iter", fontsize=font_size)
    plt.ylabel("Root Mean Squared Error", fontsize=font_size)
    plt.title("Learning curves w.r.t. the training time")
    plt.show()
    

# 11.4. Plot hypothesis and decision boundary
if let_plot:
    # Plot samples:
    plt.figure(figsize=(10, 6))
    plt.plot(X[y==2, 0], X[y==2, 1], "bo", label="Iris virginica")
    plt.plot(X[y==1, 0], X[y==1, 1], "gs", label="Iris versicolor")
    plt.plot(X[y==0, 0], X[y==0, 1], "r*", label="Iris setosa")

    # Contour plot of hypothesis function of 1 class:
    x0, x1 = np.meshgrid(
                np.linspace(0, 8, 500).reshape(-1, 1),
                np.linspace(0, 3.5, 200).reshape(-1, 1) )
    X_test = np.c_[x0.ravel(), x1.ravel()]
    y_proba = best_model.predict_proba(X_test)
    #z_hypothesis = y_proba[:, 0].reshape(x0.shape) # hypothesis of class 0: Iris setosa 
    z_hypothesis = y_proba[:, 1].reshape(x0.shape) # hypothesis of class 1: Iris versicolor
    #z_hypothesis = y_proba[:, 2].reshape(x0.shape) # hypothesis of class 2: Iris virginica
    contour = plt.contour(x0, x1, z_hypothesis, levels=25, cmap=plt.cm.Greens)
    #plt.clabel(contour, inline=1, fontsize=12)
    plt.colorbar()

    # Plot decision boundary (filled areas):
    y_predict = best_model.predict(X_test)
    z_boundary = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#f7e1e1','#e1fae1','#c8dbfa'])
    plt.contourf(x0, x1, z_boundary, cmap=custom_cmap)
    
    plt.xlabel("Petal length", fontsize=font_size)
    plt.ylabel("Petal width", fontsize=font_size)
    plt.legend(loc="upper left", fontsize=font_size)
    plt.axis([0, 7, 0, 3.5])
    plt.title("Contour plot of hypothesis of class 1: Iris versicolor", fontsize=font_size)
    plt.show()





 