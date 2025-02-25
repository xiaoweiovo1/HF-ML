import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import joblib

# load data
train_data = pd.read_csv('/data/split/train_set.csv')
test_data = pd.read_csv('/data/split/val_set.csv')

# input
X_train = train_data.iloc[:, 1:35]
y_train = train_data.iloc[:, 35]
X_test = test_data.iloc[:, 1:35]
y_test = test_data.iloc[:, 35]

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# SimpleImputer progress
imputer = SimpleImputer(strategy='mean')  
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, solver='adam',alpha=0.001, learning_rate_init=0.001, random_state=42, early_stopping=True, validation_fraction=0.1)


# train model
mlp.fit(X_train_imputed, y_train)

# save model
joblib.dump(mlp, '/class_result/best_mlp_model.pkl')

# prediction
y_pred_proba = mlp.predict_proba(X_test_imputed)[:, 1] 
y_pred = (y_pred_proba > 0.5).astype(int) 

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy: {accuracy * 100:.2f}%')


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
plt.savefig('/class_result/roc_mlp.png')


