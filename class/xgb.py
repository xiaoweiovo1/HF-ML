import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_curve, auc
# from lifelines import concordance_index
import matplotlib.pyplot as plt
import random


# load data
train_data = pd.read_csv('/data/split/train_set.csv')
test_data = pd.read_csv('/data/split/val_set.csv')

# input
X_train = train_data.iloc[:, 1:35]
y_train = train_data.iloc[:, 35]
y_time_train = train_data.iloc[:, 36]

X_test = test_data.iloc[:, 1:35]
y_test = test_data.iloc[:, 35]
y_time_test = test_data.iloc[:, 36]

# to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

# setting
params = {
    'objective': 'binary:logistic', 
    'eval_metric': 'logloss',  
    'max_depth': 8, 
    'learning_rate': 0.08,  
    'n_estimators': 100, 
    'subsample': 0.8,  
    'colsample_bytree': 0.1, 
    'seed': 42
}
num_boost_round = 1000
early_stopping_rounds = 10

# train model
bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                evals=[(dtest, 'eval')])

# prediction
y_pred_proba = bst.predict(dtest)  
y_pred = (y_pred_proba > 0.5).astype(int)  

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy: {accuracy * 100:.2f}%')

# save model
bst.save_model('/data7/qbx/qbx/model/HF/class_result/best_xgboost_model.json')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc:.3f}')

# draw ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# save ROC curve
plt.savefig('/data7/qbx/qbx/model/HF/class_result/XGB_roc_curve.png')
