import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

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

# SimpleImputer progress
imputer = SimpleImputer(strategy='mean') 
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# train Logistic Regression model
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_imputed, y_train)

# prediction
y_pred_proba = logreg.predict_proba(X_test_imputed)[:, 1]  
y_pred = (y_pred_proba > 0.5).astype(int)  

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy: {accuracy * 100:.2f}%')

# save model
import joblib
joblib.dump(logreg, '/class_result/best_logistic_model.pkl')


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

# save fig
plt.savefig('/class_result/roc_curve_logistic.png')
