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
import os   
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

def plot_model(model_class, polynomial, alphas, **model_kargs):
    # Plot data and true model
    plt.plot(X, y, "k.")
    plt.plot(X, y_no_noise, 'k-', linewidth=3, label="true model")
    
    # Learn and plot trained models
    for alpha, plot_style in zip(alphas, ("g-", "b-", "r-")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model)   ])
        model.fit(X, y)
        y_test_regul = model.predict(X_test)
        plt.plot(X_test, y_test_regul, plot_style, linewidth=2, label=r"$\alpha = $" + str(alpha))
    plt.legend(loc="upper left", fontsize=font_size-1)
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.axis([0, 3, 0, 4])


########################
# Lasso regularization #
########################
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

m = 20
np.random.seed(15);
X = 3*np.random.rand(m, 1)
y_no_noise = 1 + 0.5*X 
y = y_no_noise + np.random.randn(m, 1)/1.5
X_test = np.linspace(0, 3, 100).reshape(100, 1)

lasso_reg = Lasso(alpha=1, random_state=42)
lasso_reg.fit(X, y)

plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), random_state=42)
plt.title("Train polynomial models (degree = 10)", fontsize=font_size)
plt.show()

###############
# Elastic net #
###############
r_list = (0,0.5,1)

plt.plot(X, y, "k.")
plt.plot(X, y_no_noise, 'k-', linewidth=3, label="true model")
for r, plot_style in zip(r_list, ("g-", "b-", "r-")):
    model = ElasticNet(l1_ratio = r, alpha = 10**-7)
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("std_scaler", StandardScaler()),
        ("regul_reg", model)   ])
    model.fit(X, y)
    y_test_regul = model.predict(X_test)
    plt.plot(X_test, y_test_regul, plot_style, linewidth=2, label=r"$r = $" + str(r))
plt.legend(loc="upper left", fontsize=font_size-1)
plt.xlabel("$x_1$", fontsize=font_size)
plt.axis([0, 3, 0, 4])
