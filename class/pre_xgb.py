import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# Load the trained model
bst = xgb.Booster()
bst.load_model('/class_result/best_xgboost_model.json')

set_names = ['train', 'val', 'test']
for set_name in set_names:
    # Load validation set
    validation_data = pd.read_csv('/data/split/' + set_name + '_set.csv')
    
    # Separate features and target variable
    X_val = validation_data.iloc[:, 1:35]
    y_val = validation_data.iloc[:, 35]
    y_time_val = validation_data.iloc[:, 36]
    
    # Convert to DMatrix format (XGBoost handles missing values)
    dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan)
    
    # Make predictions
    y_pred_proba = bst.predict(dval)  # Predicted probabilities
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert to binary predictions
    y_pred_str = pd.Series(y_pred).map({0: 'Low Risk', 1: 'High Risk'})
    
    # Add prediction results to validation data
    validation_data['Predicted_Probability'] = y_pred_proba
    validation_data['Predicted_Label'] = y_pred
    
    # Save results to a new CSV file
    validation_data.to_csv('/class_result/' + set_name + '_xgb_predictions.csv', index=False)
    
    # Calculate C-index
    c_index = concordance_index(y_time_val, y_pred_proba)
    print(f'C-index: {c_index:.3f}')
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc:.3f}')
    
    # Plot and save ROC curve
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
    plt.savefig('/class_result/' + set_name + 'roc_xgb.png')
