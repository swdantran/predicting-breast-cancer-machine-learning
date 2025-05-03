# %%
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import missingno as msno 
import warnings

# %%

warnings.filterwarnings('ignore')

sns.set()
plt.style.use('ggplot')

df = pd.read_csv('breast-cancer.csv')
df.head()

# %%
# DATA PROCESSING
df.diagnosis.unique()

# %%
# M: Malignant
# B: Benign

# %%
# Supervised-> target 
# Unsupervised 
df.describe()

# %%
df.info()

# %%
# missing value
df.isna().sum()

# %%
msno.bar(df, color = 'red')

# %%
# There are no missing value in the dataset 

# %%
df['diagnosis'] = df['diagnosis'].apply(lambda val:1 if val=='M' else 0)

# %%
plt.hist(df['diagnosis'])
plt.title('Diagnosis(M=1, B=0)')
plt.show()

# %%
# EDA

# %%
# each 5 row has 6 columns
# density graph 

plt.figure(figsize=(20,15))
plotnumber=1
for column in df:
    if plotnumber<=30:
        ax = plt.subplot(5,6, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)
    plotnumber+=1

plt.tight_layout()
plt.show()

# %%
df.corr()

# %%
# heatmap
plt.figure(figsize=(20,12))
corr=df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, linewidths=1, annot=True, fmt = '.2f')

# %%
# highly correlated feature 
# multicollinearity

# %%
df.drop('id', axis=1, inplace=True)

# %%
# feature selection
corr_matrix = df.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

to_drop = [x for x in tri_df.columns if any(tri_df[x]>0.92)]

df = df.drop(to_drop, axis=1)
print(df.shape[1])

# %%
df.head()

# %%
# 32 feature reduce to 23 now

# %%
x=df.drop('diagnosis', axis=1)
y=df['diagnosis']

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=0)


# %%
# scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# %%
x_train.shape

# %%
# apply machine learning algorithm

# %%
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# %%
y_pred = log_reg.predict(x_test)

# %%
y_pred

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_train, log_reg.predict(x_train)))
log_reg_acc = accuracy_score(y_train, log_reg.predict(x_train))
print(log_reg_acc)
y_pred = log_reg.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# %%
y_pred = knn.predict(x_test)

# %%
y_pred

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_train, knn.predict(x_train)))
knn_acc = accuracy_score(y_train, knn.predict(x_train))
print(knn_acc)
y_pred = knn.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
# SVC
# Hyperparameter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
svc = SVC(probability=
          True)

parameters = {
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.1, 1.10, 15, 20]
}
grid_search = GridSearchCV(svc, parameters)
grid_search.fit(x_train, y_train)

# %%
grid_search.best_params_

# %%
grid_search.best_score_

# %%
svc = SVC(C=15, gamma=0.01, probability=True)
svc.fit(x_train, y_train)

# %%
y_pred = svc.predict(x_test)


# %%
y_pred

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_train, svc.predict(x_train)))
svc_acc = accuracy_score(y_train, svc.predict(x_train))
print(svc_acc)
y_pred = svc.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
# DT
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
parameters = {
    'criterion' :['gini', 'entropy'],
    'max_depth' :range(2,32,1),
    'min_samples_leaf' :range(1,10,1),
    'min_samples_split' :range(2,10,1),
    'splitter' :['best', 'random']
    }
grid_search_dt = GridSearchCV(dtc, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_dt.fit(x_train, y_train) 

# %%
grid_search_dt.best_params_

# %%
grid_search_dt.best_score_

# %%
dtc = DecisionTreeClassifier(criterion= 'entropy', max_depth=15, min_samples_leaf=4, min_samples_split=5, splitter= 'random')

# %%
dtc.fit(x_train, y_train)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_train, dtc.predict(x_train)))
dtc_acc = accuracy_score(y_test, dtc.predict(x_test))
print(dtc_acc)
y_pred = dtc.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 0.5, min_samples_leaf= 2, min_samples_split= 3, n_estimators= 1)
rand_clf.fit(x_train, y_train)

# %%
y_pred = rand_clf.predict(x_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_train, rand_clf.predict(x_train)))
rand_clf_acc = accuracy_score(y_test, rand_clf.predict(x_test))
print(rand_clf_acc)
y_pred = rand_clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1],
    'n_estimators': [100, 150, 180]
}
grid_search_gbc = GridSearchCV(gbc, parameters, cv = 2, n_jobs =-5, verbose= 1)
grid_search_gbc.fit(x_train, y_train)


# %%
grid_search_gbc.best_params_

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_train, gbc.predict(x_train)))
gbc_acc = accuracy_score(y_test, gbc.predict(x_test))
print(gbc_acc)
y_pred = gbc.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
from xgboost import XGBClassifier 
xgb = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.01, max_depth = 5, n_estimators =180)
xgb.fit(x_train, y_train)

# %%
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'DecisionTreeClassifier', 'Random Forest Classifier',
    'Score': [100*round(log_reg_acc,4), 100*round(knn_acc,4), 100*round(svc_acc,4), 100*round(dtc_acc,4),
                        100*round(gbc_acc,4), 100*round(xgb_acc,4)]]})
models.sort_values(by = 'Score', ascending = False)

# %%
import pickle
model = svc
pickle.dump(model, open("breast_cancer.pkl","wb"))

# %%
from sklearn import metrics
plt.figure(figsize=(8,5))
models = [
    {
        'label': 'LR',
        'model': log_reg,
    },
    {
        'label': 'DT',
        'model': dtc,
    },
    {
        'label': 'SVM',
        'model': svc,
    },
    {
        'label': 'KNN',
        'model': knn,
    },
    {
        'label': 'XGBoost',
        'model': xgb,
    },
    {
        'label': 'RF',
        'model': rand_clf,
    },
    {
        'label': 'GBDT',
        'model': gbc,
    }
]

means_roc = []
means_accuracy = [100*round(log_reg_acc,4), 100*round(dtc_acc,4), 100*round(svc_acc,4), 100*round(knn_acc,4), 100*round(rand_clf_acc,4),
                        100*round(gbc_acc,4)]
for m in models:
    model = m['model']
    model.fit(x.train, y_train)
    y_pred=model.predict(x_test)
    fpr1, tpr1, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
    auc = metrics.roc_auc_score(y_test, model.predict(x_test))
    plt.plot(fpr1, tpr1, label='%s - ROC (area = %0.2f)' % (m['label'], auc))

plt.plot([0,1], [0,1], 'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.title('ROC - Breast Cancer Prediction', fontsize=12)
plt.legend(loc = 'lower right', fontsize=12)
plt.savefig('roc_breast_cancer.jpeg', format='jpeg', dpi=400, bbox_inches='tight')
plt.show()

    




