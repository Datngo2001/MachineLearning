'''
The following code is mainly from Chap 5, GÃ©ron 2019 
https://github.com/ageron/handson-ml2/blob/master/05_support_vector_machines.ipynb

LAST REVIEW: April 2022
'''

# In[0]: IMPORTS, SETTINGS
import sklearn 
assert sklearn.__version__ >= "0.20" # sklearn â¥0.2 is required
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(42) # to output the same across runs
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
from sklearn.svm import LinearSVC 


# In[1]: LINEAR DECISION BOUNDARIES      
# 1.1. Load Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]  # use only 2 classes: setosa, versicolor
y = y[setosa_or_versicolor]

def plot_samples(subplot, with_legend=False, with_ylabel=False):
    plt.subplot(subplot)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^", label="Iris versicolor")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "go", label="Iris setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    if with_legend: plt.legend(loc="upper left", fontsize=14)
    if with_ylabel: plt.ylabel("Petal width", fontsize=14)

# 1.2. Decision boundaries of arbitrary models
x1 = np.array([0, 5.5]) # points to plot
x2_model_1 = 5*x1 - 20
x2_model_2 = x1 - 1.8

# 1.3. Train a linear SVM classifier model
# 3 implementation of linear SVM classifiers: 
# 1.
# from sklearn.svm import SVC 
# svm_clf = SVC(kernel="linear", C=np.inf):  it's SLOW
# 2.
# from sklearn.linear_model import SGDClassifier
# SGDClassifier(loss="hinge", alpha=1/(m*C)): not as fast as LinearSVC(), but works with huge datasets   
# 3.
from sklearn.svm import LinearSVC # faster than SVC on large datasets
svm_clf = LinearSVC(C=np.inf) # C: larger => 'harder margins'. loss = 'hinge': a loss of SVM
svm_clf.fit(X, y)
svm_clf.predict(X) # Predicted labels

# 1.4. Plot decision boundaries of models
# Plot arbitrary model 1:
plt.figure(figsize = [16, 5])
plot_samples(subplot='131', with_legend=True, with_ylabel=True)
plt.plot(x1, x2_model_1, "k-", linewidth=3)
plt.title("Decision boundary of model 1", fontsize=14)

# Plot arbitrary model 2:
plot_samples(subplot='132')
plt.plot(x1, x2_model_2, "k-", linewidth=3)
plt.title("Decision boundary of model 2", fontsize=14)

# Plot SVM model:
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # Plot decision boundary:
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]        
    x1 = np.linspace(xmin, xmax, 200)
    x2 = -w[0]/w[1]*x1 - b/w[1] # Note: At the decision boundary, w1*x1 + w2*x2 + b = 0 => x2 = -w1/w2 * x1 - b/w2
    plt.plot(x1, x2, "k-", linewidth=3, label="SVM")
    
    # Plot gutters of the margin:
    margin = 1/w[1]
    right_gutter = x2 + margin
    left_gutter = x2 - margin
    plt.plot(x1, right_gutter, "k:", linewidth=2)
    plt.plot(x1, left_gutter, "k:", linewidth=2)

    # Highlight samples at the gutters (support vectors):
    skipped=True
    if not skipped:
        hinge_labels = y*2 - 1 # hinge loss label: -1, 1. our label y: 0, 1
        scores = X.dot(w) + b
        support_vectors_id = (hinge_labels*scores < 1).ravel()
        svm_clf.support_vectors_ = X[support_vectors_id]      
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')

plot_samples(subplot='133')
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.title("Decision boundary of SVM model", fontsize=14)
#plt.savefig("figs/01_Decision boundaries.png")
plt.show()

# 1.5. Large vs small margins (>> see slide)

# 1.6. Support vectors (>> see slide) 
if 0:
    # Plot SVM model in a separate figure
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "go")
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis("square")
    plt.axis([0, 5.5, -1, 2.5])
    plt.title("Decision boundary of SVM model", fontsize=14)
    plt.savefig("figs/02_Linear_SVM")
    plt.show()


# [SKIP] 1.7 Sensitivity to feature scales
if 0:
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    from sklearn.svm import SVC 
    svm_clf = SVC(kernel="linear", C=100)
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(9,2.7))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")
    plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")
    plot_svc_decision_boundary(svm_clf, 0, 6)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x_1$", fontsize=20, rotation=0)
    plt.title("Unscaled", fontsize=16)
    plt.axis("square")
    #plt.axis([0, 6, 0, 90])
    plt.axis([0, 6, 45, 65])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(122)
    plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")
    plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 1][ys==0], "ms")
    plot_svc_decision_boundary(svm_clf, -2, 2)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x'_1$  ", fontsize=20, rotation=0)
    plt.title("Scaled", fontsize=16)
    plt.axis("square")
    #plt.axis([0, 20, 0, 80])
    plt.axis([-2, 2, -2, 2])
    plt.show()


# In[2]: HARD MARGIN VS SOFT MARGIN

# 2.1. Hard margin (>> see slide)

# 2.1.1. Problem 1: Only works with linearly separate data
# Add an abnormal sample 
Xo1 = np.concatenate([X, [[3.4, 1.3]]], axis=0)
yo1 = np.concatenate([y, [0]], axis=0)      
# Plot new training data
let_plot=True
if let_plot:
    plt.plot(Xo1[:, 0][yo1==1], Xo1[:, 1][yo1==1], "b^")
    plt.plot(Xo1[:, 0][yo1==0], Xo1[:, 1][yo1==0], "go")
    #plt.text(0.4, 1.8, "Impossible!", fontsize=16, color="red")
    plt.annotate("Outlier", xytext=(2.6, 1.5),
                 xy=(Xo1[-1][0], Xo1[-1][1]),
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 ha="center", fontsize=14 )
    plt.axis([0, 5.5, 0, 2])
    plt.title("Can a linear model fit this data?", color="red", fontsize=14)
    plt.show()

#%% 2.1.2. Problem 2: Sensitive to outliners 
# Add an abnormal sample 
Xo2 = np.concatenate([X, [[3.2, 0.8]]], axis=0)
yo2 = np.concatenate([y, [0]], axis=0)    
# Train and plot SVM models  
svm_clf2 = LinearSVC(C=np.Inf, max_iter=5000, random_state=42)
svm_clf2.fit(Xo2, yo2)
if let_plot:
    # Plot SVM trained without outlier 
    plt.figure(figsize = [10, 5])
    plt.subplot(1,2,1)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "go")    
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.title("SVM trained without outliner", fontsize=14 )
    plt.axis([0, 5.5, 0, 2])

    # Plot SVM trained with outlier 
    plt.subplot(1,2,2)
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "b^")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "go")
    plt.annotate("Outlier", xytext=(3.2, 0.26),
                 xy=(Xo2[-1][0], Xo2[-1][1]),
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 ha="center", fontsize=14 )
    plot_svc_decision_boundary(svm_clf2, 0, 5.5)
    plt.title("SVM trained with outliner", fontsize=14 )
    plt.axis([0, 5.5, 0, 2])
    plt.show()                


#%% 2.2. Soft margin (>> see slide)
# 2.2.1. Fit SVM models
svm_clf1 = LinearSVC(C=1, random_state=42) #, loss="hinge": standard loss for classification
svm_clf1.fit(Xo2, yo2)   
svm_clf2 = LinearSVC(C=1000, random_state=42)
svm_clf2.fit(Xo2, yo2)

# 2.2.2. Plot decision boundaries and margins
if let_plot:
    plt.figure(figsize=[10, 5])
    plt.subplot(1,2,1)
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "b^", label="Iris virginica")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "go", label="Iris versicolor")
    plt.legend(loc="upper left", fontsize=12)
    plot_svc_decision_boundary(svm_clf1, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.title("LinearSVC with C = {}".format(svm_clf1.C), fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(1,2,2)
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "b^", label="Iris virginica")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "go", label="Iris versicolor")
    plot_svc_decision_boundary(svm_clf2, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.title("LinearSVC with C = {}".format(svm_clf2.C), fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    plt.savefig("figs/03_Different C values.png")
    plt.show()


# In[3]: NONLINEAR SVM 

# 3.1. Intro (>> see slide)

# 3.2. Load non-linear data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "rs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bo")
    plt.axis(axes)
    #plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)
if let_plot:
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    #plt.savefig("figs/04_Nonlinear_data.png");
    plt.show()


# In[4]: METHOD 1 FOR NONLINEAR DATA: ADD POLINOMIAL FEATURES AND TRAIN LINEAR SVM
# 4.1. Add polinomial features and train linear svm 
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=40, random_state=42)) ])
polynomial_svm_clf.fit(X, y)

# Plot decision boundary
def plot_predictions(clf, axes, no_of_points=500):
    x0 = np.linspace(axes[0], axes[1], no_of_points)
    x1 = np.linspace(axes[2], axes[3], no_of_points)
    x0, x1 = np.meshgrid(x0, x1)
    X = np.c_[x0.ravel(), x1.ravel()]

    # Plot predicted labels (decision boundary)
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.bwr, alpha=0.12)  
    
    # Contour plot of samples' scores  
    #y_decision = clf.decision_function(X).reshape(x0.shape)
    #plt.contourf(x0, x1, y_decision, cmap=plt.cm.bwr, alpha=0.5)
    #plt.colorbar()

if let_plot:
    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()

#%% 4.2. Kernel trick for method 1: Polynomial kernel
from sklearn.svm import SVC
# NOTE: 
#   larger coef0 => the more the model is influenced by high-degree polynomials
poly_svm_1 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=.001, C=5))  ]) 
poly_svm_1.fit(X, y)

poly_svm_2 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=100, C=5))  ])
poly_svm_2.fit(X, y)

if let_plot:
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plot_predictions(poly_svm_1, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
    plt.title(r"degree={}, coef0={}, C={}".format(poly_svm_1[1].degree,poly_svm_1[1].coef0,poly_svm_1[1].C), fontsize=14)

    plt.subplot(1,2,2)
    plot_predictions(poly_svm_2, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
    plt.title(r"degree={}, coef0={}, C={}".format(poly_svm_2[1].degree,poly_svm_2[1].coef0,poly_svm_2[1].C), fontsize=14)
    plt.ylabel("")
    plt.show()


# In[5]: METHOD 2: ADD SIMILARITY FEATURES AND TRAIN 
# 5.1. Generate 1-fearture data (1-dimenstional data)
X_1D = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]).reshape(-1,1) 
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0]) # 2 classes

# 5.2. Plot Gaussian kernel graphs
def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)
def plot_kernel(X_1D,y,landmark,gamma, no_plot_points=200, xy_lim = [-4.5, 4.5, -0.1, 1.1]):  
    # Plot samples:
    plt.axhline(y=0, color='k') # Ox axis
    plt.plot(X_1D[y==0], np.zeros(4), "rs", markersize=9, label="Data samples (class 0)")
    plt.plot(X_1D[y==1], np.zeros(5), "g^", markersize=9, label="Data samples (class 1)")

    # Plot the landmark:
    plt.scatter(landmark, [0], s=200, alpha=0.5, c="orange")
    plt.annotate(r'landmark',xytext=(landmark, 0.2),
                 xy=(landmark, 0), ha="center", fontsize=14,
                 arrowprops=dict(facecolor='black', shrink=0.1)  )
    
    # Plot Gaussian kernel graph: 
    x1_plot = np.linspace(-4.5, 4.5, no_plot_points).reshape(-1,1)  
    x2_plot = gaussian_rbf(x1_plot, landmark, gamma)
    plt.plot(x1_plot, x2_plot, "b--", linewidth=2, label="Gaussian kernel")
    
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=13)
    #plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.axis(xy_lim)
    plt.title(r"Gaussian kernel with $\gamma={}$".format(gamma), fontsize=14)

# Gaussian kernel 1
landmark1 = np.array([-1.5])
gamma1 = 0.16
if let_plot:
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plt.legend(fontsize=12, loc="upper right")
    plt.show()

# Gaussian kernel 2: larger gamma, more concentrate around the landmark
landmark2 = np.array([0.26])
gamma2 = 0.51
if let_plot:
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=12, loc="upper right")
    plt.show()


#%% 5.3. Data transformation (>> see slide) 
def plot_transformed_data(X_2D,y,xy_lim=[-4.5, 4.5, -0.1, 1.1]):
    plt.axhline(y=0, color='k') # Ox
    #plt.axvline(x=0, color='k') # Oy
    plt.plot(X_2D[:, 0][y==0], X_2D[:, 1][y==0], "rs", markersize=9, label="Samples (class 0)")
    plt.plot(X_2D[:, 0][y==1], X_2D[:, 1][y==1], "g^", markersize=9, label="Samples (class 1)")

    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=14)
    plt.axis(xy_lim)
    plt.title("Data in new feature space", fontsize=14)

# 5.3.1. Use Gaussian kernel 1 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1)]
    plot_transformed_data(X_2D,y)
    plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()

# 5.3.2. Use Gaussian kernel 2 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark2, gamma2)]
    plot_transformed_data(X_2D,y)
    #plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()


# 5.3.3. Use Gaussian kernels with 2 landmarks (add 2 features)
if let_plot:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plot_kernel(X_1D,y,landmark2,gamma2)    
    plt.title("2 Gaussian kernels", fontsize=14)#plt.legend(fontsize=10, loc="upper right")

    #from mpl_toolkits.mplot3d import Axes3D 
    ax = fig.add_subplot(122, projection='3d')
    X_3D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1), 
                 gaussian_rbf(X_1D, landmark2, gamma2)]
    ax.scatter(X_3D[:, 0][y==0], X_3D[:, 1][y==0], X_3D[:, 2][y==0], 
                s=115,c="red",marker='s',label="Samples (class 0)")
    ax.scatter(X_3D[:, 0][y==1], X_3D[:, 1][y==1], X_3D[:, 2][y==1], 
                s=115,c="green",marker='^',label="Samples (class 1)")

    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$\n(similarity to lm 1)", fontsize=12)
    ax.set_zlabel("$x_3$\n(similarity to lm 2)", fontsize=12)
    plt.title("Data in new feature space", fontsize=14)
    plt.show()


#%% 5.4. How to choose landmarks? (>> see slide)


# 5.6. Kernel trick for method 2 (Gaussian kernel)
# Generate non-linear data
X, y = make_moons(n_samples=100, noise=0.22, random_state=42)

# Train 1 Gaussian SVM using Kernel trick 
Gaus_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))  ])  
Gaus_kernel_svm_clf.fit(X, y)
Gaus_kernel_svm_clf.predict(X)

# Train several Gaussian SVMs using Kernel trick 
gamma1, gamma2 = 0.1, 10
C1, C2 = 0.001, 100
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    Gaus_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("", SVC(kernel="rbf", gamma=gamma, C=C)) ])
    Gaus_kernel_svm_clf.fit(X, y)
    svm_clfs.append(Gaus_kernel_svm_clf)

# Plot boundaries by different SVMs
plt.figure(figsize=(11, 9))
for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(2,2,i+1)
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"Use Gaus. kernel with $\gamma = {}, C = {}$".format(gamma, C), fontsize=14)
    if i in (0, 1): 
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")
plt.show()

# 5.7. (>> see slide) What is the effect of: 
#   Large / small C?
#   Large / small gamma: ?


'''________________________________________________'''


# In[6]: SVM REGRESSION
# 6.1. Generata non-linear 1D data
np.random.seed(42)
m = 30 
X = 4*np.random.rand(m, 1) -2
y = (4 + 5*X**2 + X + np.random.randn(m, 1)).ravel()

# 6.2. Fit Linear SVM regressors
from sklearn.svm import LinearSVR
svm_reg1 = LinearSVR(epsilon=2, random_state=42)
svm_reg1.fit(X, y)
svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)      
svm_reg2.fit(X, y)

# 6.3. Plot the hypothesis
#def find_support_vectors(svm_reg, X, y):
#    y_pred = svm_reg.predict(X)
#    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
#    return np.argwhere(off_margin)
def plot_svm_regression(svm_reg, X, y, axes):
    # Plot model, margins
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=3, label=r"Hypothesis $\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "b--", linewidth=1, label="Margins")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "b--", linewidth=1)
    
    # Mask violated samples:
    #plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    # Plot samples:
    plt.plot(X, y, "bo")
    
    plt.axis(axes)

let_plot=True
if let_plot:
    plt.figure(figsize=(9, 5))
    plt.subplot(1,2,1)
    xylim = [-2, 2, 3, 11]
    plot_svm_regression(svm_reg1, X, y, xylim)
    # Plot epsilon:
    x1_esp = 1
    y_esp = svm_reg1.predict([[x1_esp]])
    plt.plot([x1_esp, x1_esp], [y_esp, y_esp - svm_reg1.epsilon], "k-", linewidth=2)
    plt.annotate( '', xy=(x1_esp, y_esp), xycoords='data',
            xytext=(x1_esp, y_esp - svm_reg1.epsilon),
            textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 2.5}  )
    plt.text(x1_esp+.1, y_esp-svm_reg1.epsilon/2, r"$\epsilon$ = {}".format(svm_reg1.epsilon), fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.title(r"Model trained with $\epsilon$ = {}".format(svm_reg1.epsilon), fontsize=14)
    plt.ylabel(r"$y$", fontsize=14, rotation=0)
    
    plt.subplot(1,2,2)
    plot_svm_regression(svm_reg2, X, y, xylim) 
    plt.title(r"Model trained with $\epsilon = {}$".format(svm_reg2.epsilon), fontsize=14)
    plt.savefig("figs/05_SVM_reg_epsilon");
    plt.show()

# 6.4. Which one fits the data better? (>> see slide)


#%% 6.5. Non-linear SVM regression
from sklearn.svm import SVR
# Recall: 
#   smaller epsilon ==> less data fitted (less overfitting)
#   smaller C ==> "softer" margins (less overfitting)
svm_poly_reg1 = SVR(kernel="poly", degree=8, epsilon=0.2, C=0.01, gamma="scale")
svm_poly_reg1.fit(X, y)
svm_poly_reg2 = SVR(kernel="poly", degree=8, epsilon=0.2, C=1, gamma="scale")
svm_poly_reg2.fit(X, y)
svm_poly_reg3 = SVR(kernel="poly", degree=8, epsilon=2, C=0.01, gamma="scale")
svm_poly_reg3.fit(X, y)
svm_poly_reg4 = SVR(kernel="poly", degree=8, epsilon=2, C=1, gamma="scale")
svm_poly_reg4.fit(X, y)

"""
########### Giai Thich ##############
Khi epsilon tăng lên thì độ rông của margin tăng 
sẻ làm cho model khộng đi sát với dử liệu

Khi C tăng lên thì model sẻ cho phép ít đường nằm
ngoài margin hơn từ đó làm model chình sác hơn 
"""

if let_plot:
    plt.figure(figsize=(12, 9))
    plt.subplot(2,2,1)
    xylim = [-2, 2, 3, 11]
    plot_svm_regression(svm_poly_reg1, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg1.degree, svm_poly_reg1.epsilon, svm_poly_reg1.C), fontsize=14)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    
    plt.subplot(2,2,2)
    plot_svm_regression(svm_poly_reg2, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg2.degree, svm_poly_reg2.epsilon, svm_poly_reg2.C), fontsize=14)
    
    plt.subplot(2,2,3)
    plot_svm_regression(svm_poly_reg3, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg3.degree, svm_poly_reg3.epsilon, svm_poly_reg3.C), fontsize=14)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.xlabel(r"$x_1$", fontsize=14)
    
    plt.subplot(2,2,4)
    plot_svm_regression(svm_poly_reg4, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg4.degree, svm_poly_reg4.epsilon, svm_poly_reg4.C), fontsize=14)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.show()

# 6.6. (exercise) Explain why epsilon=1 leads to wrong models (with both large and small C).
print("\n")


# In[7]: SVM OUTLIER DETECTION 
# NOTE: 
#   OneClassSVM (UNsupervised fashion):  NOT effective
#   SVM classifier (supervised fashion: class normal and class outlier): BETTER

# 7.1. Generate non-linear 1D data
np.random.seed(42)
m = 50 
x1_normal = 8*np.random.rand(m, 1) -2
x1_outlier = 0.2*np.random.rand(3, 1)
x1 = np.c_[x1_normal.T,x1_outlier.T].T
x2_normal = 4 + 3*x1_normal**2 + x1_normal + np.random.randn(m, 1)
x2_outlier = 2*np.random.rand(3, 1) +12
x2 = np.c_[x2_normal.T,x2_outlier.T].T
X = np.c_[x1,x2]
X_normal = np.c_[x1_normal,x2_normal]
y = np.c_[np.ones(x1_normal.shape).T, np.zeros(x1_outlier.shape).T].ravel()

# [NOT GOOD] 7.2.1 Train a SVM outlier detector (unsupervised fashion)
# NOTE: only use NORMAL data in training
from sklearn.svm import OneClassSVM 
#ocsvm = OneClassSVM(kernel="rbf", nu=0.001, gamma=0.1)
ocsvm = OneClassSVM(kernel='poly', degree=2,coef0=10, gamma=0.1)
ocsvm.fit(X_normal)                       

# [BETTER] 7.2.2 Train a SVM classifier (supervised fashion)
from sklearn.svm import SVC 
svc = SVC(kernel="rbf", gamma=5, C=1)
svc.fit(X,y)

# 7.3. Plot outliers
if let_plot:     
    # Plot predicted normal samples
    y_pred = ocsvm.predict(X)
    #y_pred = svc.predict(X)
    id_normal = (y_pred==1)
    plt.plot(x1[id_normal], x2[id_normal], "bo", label=r"Predicted normal samples")
    
    # Plot outliers
    id_outlier = (y_pred==-1) # -1: outlier class in ocsvm
    #id_outlier = (y_pred==0) # 0: outlier class in SVC
    plt.plot(x1[id_outlier], x2[id_outlier], "ro", label=r"Predicted outliers")      

    # Plot decision boundary
    x1_mes, x2_mes = np.meshgrid(np.linspace(min(x1), max(x1), 50), np.linspace(min(x2), max(x2), 50));
    #scores = ocsvm.decision_function(np.c_[x1_mes.ravel(), x2_mes.ravel()])
    scores = svc.decision_function(np.c_[x1_mes.ravel(), x2_mes.ravel()])
    scores = scores.reshape(x1_mes.shape)
    plt.contour(x1_mes, x2_mes, scores, levels=[0], colors="blue")

    plt.legend(fontsize=12)
    plt.show()


# In[8]: MATH BEHIND SVM
# 8.1. Load data
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
setosa_or_virginica = (y == 0) | (y == 2)
X = X[setosa_or_virginica]  # use only 2 classes: setosa, virginica
y = y[setosa_or_virginica]

# 8.2. Fit a linear SVM
svm_clf = LinearSVC(C=1000) # larger C: less regularization
svm_clf.fit(X,y);

# 8.3. Plot data and decision function surface
def plot_3D_decision_function(w, b, x1_lim, x2_lim ):
    # require: pip install pyqt5
    import matplotlib as qt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot samples
    ax.plot(X[:, 0][y==2], X[:, 1][y==2], 0, "b^")
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], 0, "go")

    # Plot surface z=0
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    ax.plot_surface(x1, x2, np.zeros(x1.shape),  color="w", alpha=0.3) #, cstride=100, rstride=100)
                                                       
    # Plot decision boundary (and margins)
    m = 1 / np.linalg.norm(w)
    x2s_boundary = -x1s*(w[0]/w[1])-b/w[1]
    ax.plot(x1s, x2s_boundary, 0, "k-", linewidth=3, label=r"Decision boundary")
    x2s_margin_1 = -x1s*(w[0]/w[1])-(b-1)/w[1]
    x2s_margin_2 = -x1s*(w[0]/w[1])-(b+1)/w[1]         
    ax.plot(x1s, x2s_margin_1, 0, "k--", linewidth=1, label=r"Margins at h=1 and -1") 
    ax.plot(x1s, x2s_margin_2, 0, "k--", linewidth=1)
     
    # Plot decision function surface
    xs = np.c_[x1.ravel(), x2.ravel()]
    dec_func = (xs .dot(w) + b).reshape(x1.shape)      
    #ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    ax.plot_surface(x1, x2, dec_func, alpha=0.3, color="r")
    ax.text(4, 1, 3, "Decision function $h$", fontsize=12)       

    ax.axis(x1_lim + x2_lim)
    ax.set_xlabel(r"Petal length", fontsize=12, labelpad=10)
    ax.set_ylabel(r"Petal width", fontsize=12, labelpad=10)
    ax.set_zlabel(r"$h$", fontsize=14, labelpad=5)
    ax.legend(loc="upper left", fontsize=12)    
w=svm_clf.coef_[0]
b=svm_clf.intercept_[0]
plot_3D_decision_function(w,b,x1_lim=[0, 5.5],x2_lim=[0, 2])
plt.show()


#%% 8.4. Slope and margin (>> see slide)
def plot_2D_decision_function(w, b, x1_lim=[-3, 3]):
    # Plot decision function 
    x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
    y = w * x1 + b
    plt.plot(x1, y, linewidth=3, color="red", label="Decision func. h")

    # Plot margins at h=1 and h=-1
    m = 1 / w
    plt.plot([-m, m], [0, 0], "ko", linewidth=3, label="Margins at h=1 & -1")
    plt.plot([m, m], [0, 1], "k--", linewidth=1)
    plt.plot([-m, -m], [0, -1], "k--", linewidth=1)
    
    plt.axis(x1_lim + [-2, 2])
    plt.xlabel(r"$x_1$", fontsize=14)
    #plt.grid()
    plt.axhline(y=0, color='k')
    #plt.axvline(x=0, color='k')
    plt.title(r"Decision func. with $w_1 = {}$".format(w), fontsize=14)

plt.figure(figsize=(9, 5))
plt.subplot(1,2,1)
plot_2D_decision_function(1, 0)
plt.ylabel(r"h = $w_1 x_1$Â + 0", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(1,2,2)
plot_2D_decision_function(0.5, 0) 
plt.show()
 
DONE = True

"""
###########
#  Cau 2  #
###########
"""
m = 100
np.random.seed(15);
X = 3*np.random.rand(m, 1)
y_no_noise = 1 + 2*X 
y = y_no_noise + np.random.randn(m, 1)/1.5

