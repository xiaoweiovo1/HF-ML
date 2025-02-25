import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the model
model_path = '/test.json'
model = xgb.Booster()
model.load_model(model_path)

# 2. Get feature importance
importance = model.get_score(importance_type='weight')  # Alternatively, use 'gain' or 'cover'

# 3. Sort by importance
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# 4. Extract features and importance scores
features, scores = zip(*sorted_importance)

# 5. Plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=list(scores), y=list(features), palette='viridis')

# 6. Add labels and title
plt.xlabel('Feature Importance (Weight)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('XGBoost Feature Importance', fontsize=14)
plt.savefig('/feature_importance.png')
