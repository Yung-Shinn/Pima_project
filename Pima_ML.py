#Import pakages
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.metrics import accuracy_score , confusion_matrix , mean_squared_error
import csv 
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import f_regression
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#%% 
# Data Proprocessing
df = pd.read_csv('pima.csv')
df.describe()
# 
y_counts = df["Outcome"].value_counts()
print(y_counts)

#%% Remove outlier
# model buliding
iforest = IsolationForest(n_estimators=300,  #The number of base estimators(trees)
                          max_samples='auto', #The number of samples to train
                          contamination=0.05,  # the proportion of outliers in the data set.
                          max_features=3, #The number of features to train
                          n_jobs=-1, 
                          random_state=1)

df_pred = iforest.fit_predict(df)
df_scores = iforest.decision_function(df)
df_anomaly_label = df_pred
pima_outlier = df[df_anomaly_label==-1]

print('--- Observations found as outliers using isolation forest-----')
print(pima_outlier)
print(pima_outlier.shape)

df_cleaned = df[df_anomaly_label==1]

#%% Data Split
X = df_cleaned.iloc[:,:8]
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)
X_train.shape, X_test.shape
#%% 特徵萃取
#principal component analysis
#pca_pima = PCA(n_components=8) #8個主成分
#pca_pima.fit_transform(X_train)

#eigen_value = pd.DataFrame(np.round(pca_pima.explained_variance_), columns=['eigen value']) #變異量,越大越好
#print('eigen value:\n',eigen_value)

#pima_var = np.round(pca_pima.explained_variance_ratio_,4)*100 #8個主成分各自解釋了多少%變異
#pima_vardf = pd.DataFrame(pima_var)
#pima_vardf.columns = ['explain var']
#print("\n explained variances(%):\n",pima_vardf)

#pima_cumvar = np.cumsum(pima_var) #8個主成分累計解釋變異
#pima_cumvardf = pd.DataFrame(pima_cumvar)
#pima_cumvardf.columns = ['cum explain var']
#print("\n accumulative explained variances(%):\n",pima_cumvardf)

#plt.plot(pima_cumvar)
#plt.xlabel('components')
#plt.ylabel('cumulative explained variance')
#只需要4個主成分就能解釋99.5%以上的變異

#pca_fin = PCA(n_components=4)
#X_train_pca = pca_fin.fit_transform(X_train) 
#X_test_pca = pca_fin.transform(X_test)

#%% LogisticRegression
# parameter grid
param_grid_LR = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

clf = LogisticRegression()

grid_search = GridSearchCV(clf, param_grid_LR, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("最佳参数: ", grid_search.best_params_)
print("最高準確率: ", grid_search.best_score_)

# Build classifier
LR =LogisticRegression(penalty = 'l1', C = 10, solver = 'liblinear')
LR_pima = LR.fit(X_train, y_train)

# Predict the train subset
train_pred_LR = LR_pima.predict(X_train)
train_conf_LR=pd.crosstab( y_train, train_pred_LR, rownames=['real'],colnames=['pred'])
print(train_conf_LR)
print(classification_report(y_train, train_pred_LR))

# Predict the test subset
test_pred_LR = LR_pima.predict(X_test)
test_conf_LR=pd.crosstab(y_test, test_pred_LR, rownames=['real'],colnames=['pred'])
print(test_conf_LR)
print(classification_report(y_test, test_pred_LR))

#%% KNN
#選擇K的個數
k_range = np.arange(2, 10)
accur = []
table=[]

for i in k_range:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn_clf = knn.fit(X_train, y_train)
    test_pred = knn_clf.predict(X_train)
    accu = metrics.accuracy_score(y_train, test_pred)
    accur.append(accu)
table=pd.Series(accur) 
print("選擇K的個數\n\n",table)

best_k = accur.index(max(accur)) + k_range[0]
print("最佳K的個數:",best_k)
plt.scatter(k_range, accur)
plt.title('Best numbers of K')  
plt.xlabel('Numbers of K') 
plt.ylabel('Accuracy') 
plt.show()

# Build classifier
knn = KNeighborsClassifier(n_neighbors = best_k)
knn_pima = knn.fit(X_train, y_train)

# Predict the train subset
train_pred_KNN = knn_pima.predict(X_train)
train_conf_KNN=pd.crosstab( y_train, train_pred_KNN, rownames=['real'],colnames=['pred'])
print(train_conf_KNN)
print(classification_report(y_train, train_pred_KNN))

# Predict the test subset
test_pred_KNN = knn_pima.predict(X_test)
test_conf_KNN=pd.crosstab(y_test, test_pred_KNN, rownames=['real'],colnames=['pred'])
print(test_conf_KNN)
print(classification_report(y_test, test_pred_KNN))

#%%
##RandomForest##

# parameter grid
params_RF = {'n_estimators':list(range(50,151,20)), #The number of trees in the forest.
             'max_depth':list(range(2,5)),  #The maximum depth of the tree.
             'max_features':list(range(2,5)), #The number of features to consider when looking for the best split
             'min_samples_split':list(range(10,31,5)), #The minimum number of samples required to split an internal node
             'min_samples_leaf':list(range(5,21,5))} #The minimum number of samples required to be at a leaf node.

# Grid Search in RF
tunnRF = GridSearchCV(RandomForestClassifier(random_state=(123)), params_RF, cv = 5)
tunnRF.fit(X_train,y_train)
    
print('\nBest parameters :',tunnRF.best_params_)

# Build classifier
rf = RandomForestClassifier(criterion = 'gini',
                            n_estimators = tunnRF.best_estimator_.n_estimators,
                            max_depth = tunnRF.best_estimator_.max_depth,
                            max_features = tunnRF.best_estimator_.max_features, 
                            min_samples_split = tunnRF.best_estimator_.min_samples_split,
                            min_samples_leaf = tunnRF.best_estimator_.min_samples_leaf)
rf_pima = rf.fit(X_train, y_train)

# Get importance of feature with sorting
pima_imp = pd.DataFrame({'Feature': X_train.columns,
                         'Importance':rf_pima.feature_importances_})
print('\nFeature importance:\n',pima_imp.sort_values(by=['Importance'],ascending=False))

# Predict the train subset
train_pred_rf = rf_pima.predict(X_train)
train_conf_rf = pd.crosstab(y_train, train_pred_rf ,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training:\n',train_conf_rf)
print(classification_report(y_train, train_pred_rf))

# Predict the test subset
test_pred_rf = rf_pima.predict(X_test)
test_conf_rf = pd.crosstab(y_test, test_pred_rf ,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training:\n',test_conf_rf)
print(classification_report(y_test, test_pred_rf))

#%%
# XGB

# parameter grid
params_XGB = {'n_estimators':list(range(50,151,20)), #The number of trees in the forest.
                  'max_depth':list(range(2,5)),  #The maximum depth of the tree.
                  'learning_rate':[0.1,0.05,0.01], #Step size shrinkage used in update to prevents overfitting.
                  'min_child_weight':[0.1,0.3,0.5], #The minimum sum of weights of all observations required in a child (not number of observations which is min_samples_leaf. ).
                  'colsample_bytree':[0.5,0.7,0.9]} #Similar to max_features. Denotes the fraction of columns to be randomly samples for each tree.
tunnXgb = GridSearchCV(XGBClassifier(eval_metric='error', random_state=(124)), params_XGB, cv = 5, )
tunnXgb.fit(X_train, y_train)

# Get Best parameter
print('\nBest parameters :',tunnXgb.best_params_)

# Build classifier
XGB = XGBClassifier(n_estimators=tunnXgb.best_estimator_.n_estimators,
                    learning_rate=tunnXgb.best_estimator_.learning_rate,
                    max_depth=tunnXgb.best_estimator_.max_depth,
                    min_child_weight=tunnXgb.best_estimator_.min_child_weight,
                    colsample_bytree=tunnXgb.best_estimator_.colsample_bytree,
                    eval_metric='error')
xgb_pima = XGB.fit(X_train, y_train)

# Get importance of feature with sorting
pima_imp_xgb = pd.DataFrame({'Feature':X_train.columns,'Importance':xgb_pima.feature_importances_})
print('\nFeature importance:\n', pima_imp_xgb.sort_values(by=['Importance'],ascending=False))

# Predict the train subset
train_pred_xgb = xgb_pima.predict(X_train)
train_conf_xgb = pd.crosstab(y_train, train_pred_xgb ,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training:\n',train_conf_xgb)
print(classification_report(y_train, train_pred_xgb))

# Predict the test subset
test_pred_xgb = xgb_pima.predict(X_test)
test_conf_xgb = pd.crosstab(y_test, test_pred_xgb ,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training:\n',test_conf_xgb)
print(classification_report(y_test, test_pred_xgb))

#%%
# 績效比較

LR_test_accuracy = metrics.accuracy_score(y_test, test_pred_LR)
LR_test_recall = metrics.recall_score(y_test, test_pred_LR, pos_label= 1, average='binary')
LR_test_precision = metrics.precision_score(y_test, test_pred_LR, pos_label= 1, average='binary')
LR_test_fmeasure = metrics.f1_score(y_test, test_pred_LR, pos_label= 1, average='binary')

KNN_test_accuracy = metrics.accuracy_score(y_test, test_pred_KNN)
KNN_test_recall =metrics.recall_score(y_test, test_pred_KNN, pos_label= 1, average='binary')
KNN_test_precision =metrics.precision_score(y_test, test_pred_KNN, pos_label= 1, average='binary')
KNN_test_fmeasure =metrics.f1_score(y_test, test_pred_KNN, pos_label= 1, average='binary')

rf_test_accuracy= metrics.accuracy_score(y_test, test_pred_rf)
rf_test_recall= metrics.recall_score(y_test, test_pred_rf, pos_label= 1, average='binary')
rf_test_precision= metrics.precision_score(y_test, test_pred_rf, pos_label= 1, average='binary')
rf_test_fmeasure= metrics.f1_score(y_test, test_pred_rf, pos_label= 1, average='binary')

xgb_test_accuracy= metrics.accuracy_score(y_test, test_pred_xgb)
xgb_test_recall= metrics.recall_score(y_test, test_pred_xgb, pos_label= 1, average='binary')
xgb_test_precision= metrics.precision_score(y_test, test_pred_xgb, pos_label= 1, average='binary')
xgb_test_fmeasure= metrics.f1_score(y_test, test_pred_xgb, pos_label= 1, average='binary')

print('Pima data')
print('Accuracy of Using Logistic Regression Classifier  =', LR_test_accuracy)
print('Accuracy of Using KNN Classifier  =', KNN_test_accuracy)
print('Accuracy of Using Random Forest Classifier  =', rf_test_accuracy)
print('Accuracy of Using XGB Classifier  =', xgb_test_accuracy)

print('Recall of Using Decision Tree Classifier  =', LR_test_recall)
print('Recall of Using KNN Classifier  =', KNN_test_recall)
print('Recall of Using Random Forest Classifier  =', rf_test_recall)
print('Recall of Using XGB Classifier  =', xgb_test_recall)

print('Precision of Using Decision Tree Classifier  =', LR_test_precision)
print('Precision of Using KNN Classifier  =', KNN_test_precision)
print('Preciison of Using Random Forest Classifier  =', rf_test_precision)
print('Preciison of Using XGB Classifier  =', xgb_test_precision)

print('Fmeasure of Using Decision Tree Classifier  =', LR_test_fmeasure)
print('Fmeasure of Using KNN Classifier  =', KNN_test_fmeasure)
print('Fmeasure of Using Random Forest Classifier  =', rf_test_fmeasure)
print('Fmeasure of Using XGB Classifier  =', xgb_test_fmeasure)


df_performance = {
    'Model': ['Logistic Regression', 'KNN', 'Random Forest', 'XGB'],
    'Accuracy': [LR_test_accuracy, KNN_test_accuracy, rf_test_accuracy, xgb_test_accuracy],
    'Recall': [LR_test_recall, KNN_test_recall, rf_test_recall, xgb_test_recall],
    'Precision': [LR_test_precision, KNN_test_precision, rf_test_precision, xgb_test_precision],
    'F1 Score': [LR_test_fmeasure, KNN_test_fmeasure, rf_test_fmeasure, xgb_test_fmeasure]
}


pima_performance = pd.DataFrame(df_performance)
print(pima_performance)


#%%
# ROC/AUC

pima_array = y_test.values

pima_y_predprob_LR = LR_pima.predict_proba(X_test)
fpr_LR, tpr_LR, thr_LR = roc_curve(pima_array, pima_y_predprob_LR[:,1], pos_label = 1)
roc_auc_LR = auc(fpr_LR, tpr_LR)
print('\nArea under curve = ', roc_auc_LR)

pima_y_predprob_KNN = knn_pima.predict_proba(X_test)
fpr_KNN, tpr_KNN, thr_KNN = roc_curve(pima_array, pima_y_predprob_KNN[:,1], pos_label = 1)
roc_auc_KNN = auc(fpr_KNN, tpr_KNN)
print('\nArea under curve = ', roc_auc_KNN)

pima_y_predprob_rf = rf_pima.predict_proba(X_test)
fpr_rf, tpr_rf, thr_rf = roc_curve(pima_array, pima_y_predprob_rf[:,1], pos_label = 1)
roc_auc_rf = auc(fpr_rf, tpr_rf)
print('\nArea under curve = ', roc_auc_rf)

pima_y_predprob_xgb = xgb_pima.predict_proba(X_test)
fpr_xgb, tpr_xgb, thr_xgb = roc_curve(pima_array, pima_y_predprob_xgb[:,1], pos_label = 1)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
print('\nArea under curve = ', roc_auc_xgb)

plt.figure()
plt.figure(figsize=(10,10))
plt.plot(fpr_LR, tpr_LR, color='blue',lw=2, label='ROC curve_LR (AUC = %.2f)'%roc_auc_LR) #lw = linewidth
plt.plot(fpr_KNN, tpr_KNN, color='red',lw=2, label='ROC curve_KNN (AUC = %.2f)'%roc_auc_KNN)
plt.plot(fpr_rf, tpr_rf, color='green',lw=2, label='ROC curve_RF (AUC = %.2f)'%roc_auc_rf)
plt.plot(fpr_xgb, tpr_xgb, color='black',lw=2, label='ROC curve_XGB (AUC = %.2f)'%roc_auc_xgb)

##Title and label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc = 'lower right')
plt.show()









