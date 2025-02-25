import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc
from lifelines.utils import concordance_index  # For calculating C-index
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # For imputing missing values

# 1. Load the trained model
model = joblib.load('/class_result/best_mlp_model.pkl')

set_names = ['train', 'val', 'test']
for set_name in set_names:
    # 2. Load test data
    test_data = pd.read_csv('/data/split/' + set_name + '_set.csv')
    
    # 3. Prepare the test data (assuming same columns as training data)
    X_test = test_data.iloc[:, 1:35]  # Feature columns
    y_test = test_data.iloc[:, 35]    # Target variable (survival status)

    # 4. Impute missing values
    imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean
    X_test_imputed = imputer.fit_transform(X_test)  # Impute test data

    # 5. Standardize the data (scale features)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_imputed)  # Standardize imputed data

    # 6. Make predictions and get predicted probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Get probabilities for the positive class
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary predictions
    y_pred_str = pd.Series(y_pred).map({0: 'Low Risk', 1: 'High Risk'})
    
    # 7. Save predictions and probabilities to new columns in the test data
    test_data['predicted_label'] = y_pred
    test_data['predicted_proba'] = y_pred_proba

    # Save prediction results to a new CSV file
    test_data.to_csv('/class_result/' + set_name + '_mlp.csv', index=False)

    # 8. Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc:.3f}')

    # 9. Calculate C-index
    y_time_test = test_data.iloc[:, 36]  # Survival time column
    c_index = concordance_index(y_time_test, y_pred_proba)
    print(f'C-index: {c_index:.3f}')

    # 10. Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Save ROC curve image
    plt.savefig('/class_result/' + set_name + 'roc_mlp.png')

    # Close the plot to avoid displaying
    plt.close()
