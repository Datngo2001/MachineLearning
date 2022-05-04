# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:07:29 2022

@author: ngomi
"""

'''
The following code is mainly from Chap 3, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb

LAST REVIEW: March 2022
'''


# In[0]: IMPORTS
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import joblib # Note: require sklearn v0.22+ (to update sklearn: pip install -U scikit-learn ). For old version sklearn: from sklearn.externals import joblib 
from sklearn.linear_model import SGDClassifier   
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[1]: MNIST DATASET
# 1.1. Load MNIST  
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

# 1.3. Plot a digit image   
def plot_(data, label = 'unspecified', showed=True):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("Name: " + str(label))
    #plt.axis("off")
    if showed:
        plt.show()
sample_id = 0
plot_(X_train[sample_id], y_train[sample_id])


# In[2]: TRAINING A BINARY CLASSIFIER (just two classes, 5 and not-5)               
# 2.1. Create label array: True for 5s, False for other digits.
y_train_0 = (y_train == 0) 
y_test_0 = (y_test == 0)

# 2.2. Try Stochastic Gradient Descent (SGD) classifier
# Note 1: In sklearn, SGDClassifier train linear classifiers using SGD, 
#         depending on the loss, eg. ‘hinge’ (default): linear SVM, ‘log’: logistic regression, etc.
# Note 2: SGD takes 1 datum at a time, hence well suited for online learning. 
#         It also able to handle very large datasets efficiently
sgd_clf = SGDClassifier(random_state=42) # set random_state to reproduce the result
# Train: 
# Warning: takes time for new run!
new_run = True
if new_run == True:
    sgd_clf.fit(X_train, y_train_0)
    joblib.dump(sgd_clf,'saved_var/sgd_clf_binary')
else:
    sgd_clf = joblib.load('saved_var/sgd_clf_binary')

# Predict a sample:
sample_id = 1000
plot_(X_train[sample_id], label=y_train[sample_id])
sgd_clf.predict([X_train[sample_id]])
#y_train_0[sample_id]

#RANDOM FOREST
randomForest = True
rfc = RandomForestClassifier(n_estimators=70, oob_score=True, random_state=101)
if randomForest:
  rfc.fit(X_train, y_train_0)
  joblib.dump(rfc,'saved_var/rfc_binary')
else:
  rfc = joblib.load('saved_var/rfc_binary')
rfc.predict([X_train[sample_id]])

# In[3]: PERFORMANCE MEASURES 
# 3.1. Accuracy (with cross-validation) of SGDClassifier 
from sklearn.model_selection import cross_val_score
# Warning: takes time for new run! 
if new_run == True:
    accuracies = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy")
    joblib.dump(accuracies,'saved_var/sgd_clf_binary_acc')
else:
    accuracies = joblib.load('saved_var/sgd_clf_binary_acc')

# 3.2. Accuracy of a dump classifier
# Note: We are having an IMBALANCED data, hence accuracy is not useful!
from sklearn.base import BaseEstimator
class DumpClassifier(BaseEstimator): # always return False (not-5 label)
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
no_5_model = DumpClassifier()
cross_val_score(no_5_model, X_train, y_train_0, cv=3, scoring="accuracy")
# Note: 
#   >90% accuracy, due to only about 10% of the images are 5s.
#   IMBALANCED (or skewed) datasets: some classes are much more frequent than others.

# 3.3. Confusion matrix (better for imbalanced data)
# Info: number of times the classifier "confused" b/w samples of classes
from sklearn.model_selection import cross_val_predict 
# Warning: takes time for new run! 
if new_run == True:
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3)
    joblib.dump(y_train_pred,'saved_var/y_train_pred')
    y_train_pred_dump = cross_val_predict(no_5_model, X_train, y_train_0, cv=3)
    joblib.dump(y_train_pred,'saved_var/y_train_pred_dump')  
else:
    y_train_pred = joblib.load('saved_var/y_train_pred')
    y_train_pred_dump = joblib.load('saved_var/y_train_pred_dump')

conf_mat = confusion_matrix(y_train_0, y_train_pred) # row: actual class, column: predicted class. 
print(conf_mat)
conf_mat_dump = confusion_matrix(y_train_0, y_train_pred_dump) # row: actual class, column: predicted class. 
print(conf_mat_dump)
# Perfect prediction: zeros off the main diagonal 
y_train_perfect_predictions = y_train_0  # pretend we reached perfection
confusion_matrix(y_train_0, y_train_perfect_predictions)


# 3.4. Precision and recall (>> see slide)
from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_0, y_train_pred))
print(recall_score(y_train_0, y_train_pred))
print(precision_score(y_train_0, y_train_pred_dump))
print(recall_score(y_train_0, y_train_pred_dump))


# 3.5. F1-score 
# Info: F1-score is the harmonic mean of precision and recall. 1: best, 0: worst.
# F1 = 2 × precision × recall / (precision + recall)
f1_score(y_train_0, y_train_pred)


# 3.6. Precision/Recall tradeoff (>> see slide) 
# 3.6.1. Try classifying using some threshold (on score computed by the model)  
sample_id = 11
y_score = sgd_clf.decision_function([X_train[sample_id]]) # score by the model
threshold = 0
y_some_digit_pred = (y_score > threshold)
y_train_0[sample_id]
# Raising the threshold decreases recall
threshold = 10000
y_some_digit_pred = (y_score > threshold)  

# 3.6.2. Precision, recall curves wrt to thresholds 
# Get scores of all intances
# Warning: takes time for new run! 
if new_run == True:
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3, method="decision_function")
    joblib.dump(y_scores,'saved_var/y_scores')
else:
    y_scores = joblib.load('saved_var/y_scores')

# Plot precision,  recall curves
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
let_plot = True
if let_plot:
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend() 
    plt.grid(True)
    plt.xlabel("Threshold")   

# Plot a threshold
thres_value = 1000
thres_id = np.min(np.where(thresholds >= thres_value))
precision_at_thres_id = precisions[thres_id] 
recall_at_thres_id = recalls[thres_id] 
if let_plot:
    plt.plot([thres_value, thres_value], [0, precision_at_thres_id], "r:")    
    plt.plot([thres_value], [precision_at_thres_id], "ro")                            
    plt.plot([thres_value], [recall_at_thres_id], "ro")            
    plt.text(thres_value+500, 0, thres_value)    
    plt.text(thres_value+500, precision_at_thres_id, np.round(precision_at_thres_id,3))                            
    plt.text(thres_value+500, recall_at_thres_id, np.round(recall_at_thres_id,3))     
    plt.show()

# 3.6.3. Precision vs recall curve (Precision-recall curve)
if let_plot:         
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.title("Precision-recall curve (PR curve)")
    plt.show()


# 3.7. Receiver operating characteristic (ROC) curve 
# Info: another common measure for binary classifiers. 
# ROC curve: the True Positive Rate (= recall) against 
#   the False Positive Rate (= no. of false postives / total no. of actual negatives).
#   FPR is the ratio of negative instances that are incorrectly classified as positive.
# NOTE: 
#   Tradeoff: the higher TPR, the more FPR the classifier produces.
#   Good classifier goes toward the top-left corner.

# 3.7.1. Compute FPR, TPR for the SGDClassifier
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)

# 3.7.2. Compute FPR, TPR for a random classifier (make prediction randomly)
from sklearn.dummy import DummyClassifier
dmy_clf = DummyClassifier(strategy="uniform")
y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_0, cv=3, method="predict_proba")
y_scores_dmy = y_probas_dmy[:, 1]
fprr, tprr, thresholdsr = roc_curve(y_train_0, y_scores_dmy)

# 3.7.3. Plot ROC curves
if let_plot:
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot(fprr, tprr, 'k--') # random classifier
    #plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal: random classifier
    plt.legend(['SGDClassifier','Random classifier'])
    plt.grid(True)        
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate (Recall)')    
    plt.show()

# 3.8. Compute Area under the curve (AUC) for ROC
# Info: 
#   A random classifier: ROC AUC = 0.5.
#   A perfect classifier: ROC AUC = 1.
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_train_0, y_scores)
    
# 3.9. ROC vs PR curve: when to use?
#   PR curve: focus the false positives (ie. u want high precision)
#   ROC: focus the false negatives (ie. u want high recall)
print('\n')


'''______DONE WEEK 05______'''


# In[5]: MULTICLASS CLASSIFICATION (>> see slide)
# 5.1. Try SGDClassifier 
# Info: SkLearn automatically runs OvA when you try to 
#       use a binary classifier for a multiclass classification.
# Warning: takes time for new run!
sample_id = 2
plot_(X_train[sample_id], y_train[sample_id])

new_run = False 
if new_run == True:
    sgd_clf.fit(X_train, y_train) # y_train, not y_train_0
    joblib.dump(sgd_clf,'saved_var/sgd_clf_multi')
else:
    sgd_clf = joblib.load('saved_var/sgd_clf_multi')
# Try prediction
sgd_clf.predict([X_train[sample_id]])
y_train[sample_id]
# To see scores from classifers
sgd_clf.classes_
sample_scores = sgd_clf.decision_function([X_train[sample_id]]) 
sample_scores
class_with_max_score = np.argmax(sample_scores)
class_with_max_score

#RANDOM FOREST
new_run = False 
rfc = RandomForestClassifier(n_estimators=70, oob_score=True, random_state=101)
if new_run == True:
  rfc.fit(X_train, y_train)
  joblib.dump(rfc,'saved_var/rfc_binary')
else:
  rfc = joblib.load('saved_var/rfc_binary')
  
rfc.predict([X_train[sample_id]])
y_train[sample_id]


# 5.2. Force sklearn to run OvO (OneVsOneClassifier) or OvA (OneVsRestClassifier)
from sklearn.multiclass import OneVsRestClassifier
# Warning: takes time for new run! 
ova_clf = OneVsRestClassifier(SGDClassifier(random_state=42))
new_run = False 
if new_run == True:
    ova_clf.fit(X_train, y_train)
    joblib.dump(ova_clf,'saved_var/ova_clf')
else:
    ova_clf = joblib.load('saved_var/ova_clf')
len(ova_clf.estimators_)
ova_clf.classes_
sample_scores = ova_clf.decision_function([X_train[sample_id]]) 
sample_scores 
class_with_max_score = np.argmax(sample_scores)
class_with_max_score

from sklearn.multiclass import OneVsOneClassifier
# Warning: takes time for new run! 
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
new_run = False 
if new_run == True:
    ovo_clf.fit(X_train, y_train)
    joblib.dump(ovo_clf,'saved_var/ovo_clf')
else:
    ovo_clf = joblib.load('saved_var/ovo_clf')
len(ovo_clf.estimators_) 
ovo_clf.classes_
sample_scores = ovo_clf.decision_function([X_train[sample_id]]) 
sample_scores
class_with_max_score = np.argmax(sample_scores)
class_with_max_score
 

# In[6]: EVALUATE CLASSIFIERS
# 6.1. SGDClassifier  
# Warning: takes time for new run! 
sgd_acc=0;
new_run = True
if new_run == True:
    sgd_acc = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    joblib.dump(sgd_acc,'saved_var/sgd_acc_multi')
else:
    sgd_acc = joblib.load('saved_var/sgd_acc_multi')
# 6.2. RandomForestClassifier  
# Warning: takes time for new run! 
forest_acc=0;
if new_run == True:
    forest_acc = cross_val_score(rfc, X_train, y_train, cv=3, scoring="accuracy")
    joblib.dump(forest_acc,'saved_var/forest_acc_multi')
else:
    forest_acc = joblib.load('saved_var/forest_acc_multi')


# In[7]: SCALE FEATURES AND EVALUATE CLASSIFIERS AGAIN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# 7.1. SGDClassifier (benefited from feature scaling)
# Warning: takes time for new run! 
new_run = True
sgd_acc_after_scaling=0;
if new_run == True:
    sgd_acc_after_scaling = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=4)
    joblib.dump(sgd_acc_after_scaling,'saved_var/sgd_acc_after_scaling')
else:
    sgd_acc_after_scaling = joblib.load('saved_var/sgd_acc_after_scaling')
# 7.2. RandomForestClassifier  
# Warning: takes time for new run! 
new_run = True
forest_acc_after_scaling=0;
if new_run == True:
    forest_acc_after_scaling = cross_val_score(rfc, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=4)
    joblib.dump(forest_acc_after_scaling,'saved_var/forest_acc_after_scaling')
else:
    forest_acc_after_scaling = joblib.load('saved_var/forest_acc_after_scaling')


# In[8]: ERROR ANALYSIS 
# NOTE: Here we skipped steps (eg. trying other data preparation options, hyperparameter tunning...)
#       Assumming that we found a promissing model, and are trying to improve it.
#       One way is to analyze errors it made.

# 8.1. Plot confusion matrix
# Warning: takes time for new run! 
y_train_pred=0;
new_run = False
if new_run == True:
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    joblib.dump(y_train_pred,'saved_var/y_train_pred_step8')
else:
    y_train_pred = joblib.load('saved_var/y_train_pred_step8')
conf_mat = confusion_matrix(y_train, y_train_pred) # row: actual class, col: prediction
let_plot = True;
if let_plot:
    plt.matshow(conf_mat, cmap=plt.cm.seismic)
    plt.xlabel("Prediction")
    plt.ylabel("Actual class")
    plt.colorbar()
    plt.savefig("figs/confusion_matrix_plot")
    plt.show()

# 8.2. Plot error-only confusion matrix 
# Convert no. of intances to rates
row_sums = conf_mat.sum(axis=1, keepdims=True)
norm_conf_mat = conf_mat / row_sums
# Replace rates on diagonal (correct classifitions) by zeros    
if let_plot:
    np.fill_diagonal(norm_conf_mat, 0)
    plt.matshow(norm_conf_mat,cmap=plt.cm.seismic)
    plt.xlabel("Prediction")
    plt.ylabel("Actual class")
    plt.colorbar()
    plt.savefig("figs/confusion_matrix_errors_plot", tight_layout=False)
    plt.show()


# 8.4. Plot examples of 3s and 5s
def plot_s(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

class_A = 2
class_B = 4
X_class_AA = X_train[(y_train == class_A) & (y_train_pred == class_A)]
X_class_AB = X_train[(y_train == class_A) & (y_train_pred == class_B)]
X_class_BA = X_train[(y_train == class_B) & (y_train_pred == class_A)]
X_class_BB = X_train[(y_train == class_B) & (y_train_pred == class_B)] 
    
if let_plot:
    plt.figure(figsize=(6,7))
    plt.subplot(221); plot_s(X_class_AA[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_A) + ", Predicted: " + str(class_A))
    plt.subplot(222); plot_s(X_class_AB[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_A) + ", Predicted: " + str(class_B))
    plt.subplot(223); plot_s(X_class_BA[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_B) + ", Predicted: " + str(class_A))
    plt.subplot(224); plot_s(X_class_BB[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_B) + ", Predicted: " + str(class_B))
    plt.show()

 
# 9.2. Plot error-only confusion matrix 
# Convert no. of intances to rates
row_sums = conf_mat.sum(axis=1, keepdims=True)
norm_conf_mat = conf_mat / row_sums
# Replace rates on diagonal (correct classifitions) by zeros    
if let_plot:
    np.fill_diagonal(norm_conf_mat, 0)
    plt.matshow(norm_conf_mat,cmap=plt.cm.seismic)
    plt.xlabel("Prediction")
    plt.ylabel("Actual class")
    plt.colorbar()
    #plt.savefig("figs/confusion_matrix_errors_plot_RF", tight_layout=False)
    plt.show()

# 9.3. Plot examples of 4s vs misclassed 8s, 3s vs misclassed 2s
if let_plot:
    class_A = 4
    class_B = 8
    X_class_AB = X_train[(y_train == class_A) & (y_train_pred == class_B)]
    plt.subplot(121); 
    plot_s(X_class_AB[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_A) + ", Predicted: " + str(class_B))

    class_A = 3
    class_B = 2
    X_class_AB = X_train[(y_train == class_A) & (y_train_pred == class_B)]
    plt.subplot(122); 
    plot_s(X_class_AB[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_A) + ", Predicted: " + str(class_B))
    plt.show()
            

# In[10]: MULTILABEL CLASSIFICATION (Multi-[binary] label) 
# Info: (>> see slide) 
# 10.1. Create multilabel labels
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

# 10.2. Try KNeighborsClassifier    
# Note: KNeighborsClassifier supports multilabel classification. Not all classifiers do. 
from sklearn.neighbors import KNeighborsClassifier
# Warning: takes time for new run! 
knn_clf = KNeighborsClassifier()
if new_run == True:
    knn_clf.fit(X_train, y_multilabel)
    joblib.dump(knn_clf,'saved_var/knn_clf')
else:
    knn_clf = joblib.load('saved_var/knn_clf')
# Try prediction
sample_id = 0;
knn_clf.predict([X_train[sample_id]])
y_multilabel[sample_id]
 
# 10.3. Evaluate a multilabel classifier
# Note: many ways to do this, e.g., measure the F1 score for each individual label then compute the average score
# WARNING: may take HOURS for a new run! 
y_train_knn_pred = 0;
if new_run == True:
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    joblib.dump(y_train_knn_pred,'saved_var/y_train_knn_pred')
else:
    y_train_knn_pred = joblib.load('saved_var/y_train_knn_pred')
f1_score(y_multilabel, y_train_knn_pred, average="macro") # macro: unweighted mean, weighted: average weighted by support (no. of true instances for each label)

 

# In[11]: MULTIOUTPUT CLASSIFICATION (>> see slide) 
# 11.1. Add noise to data
# Create noisy features
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
# Labels now are clear images
y_train_mod = X_train
y_test_mod = X_test
# Plot a sample and its label
if let_plot:
    sample_id = 0
    plt.subplot(121); 
    plot_(X_train_mod[sample_id],str(y_train[sample_id])+" (noisy FEATURE)",showed=False)
    plt.subplot(122); 
    plot_(y_train_mod[sample_id],str(y_train[sample_id])+" (LABEL)",showed=True)

# 11.2. Training
# Warning: takes time for a new run! 
if new_run == True:
    knn_clf.fit(X_train_mod, y_train_mod)
    joblib.dump(knn_clf,'saved_var/knn_clf_multioutput')
else:
    knn_clf = joblib.load('saved_var/knn_clf_multioutput')

# 11.3. Try predition
sample_id = 12
clean_digit = knn_clf.predict([X_test_mod[sample_id]])    
if let_plot:
    plt.figure(figsize=[12,5])
    plt.subplot(131); 
    plot_(X_test_mod[sample_id],str(y_test[sample_id])+" (input SAMPLE)",showed=False)
    plt.subplot(132); 
    plot_(clean_digit,str(y_test[sample_id])+" (PREDICTION)",showed=False)
    plt.subplot(133); 
    plot_(y_test_mod[sample_id],str(y_test[sample_id])+" (LABEL)",showed=True)


# End of our classification tour.
end = True

 

 
