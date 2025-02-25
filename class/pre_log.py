import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc
from lifelines.utils import concordance_index  # 用于计算C-index
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# 加载训练好的Logistic Regression模型
model_path = '/class_result/best_logistic_model.pkl'
logreg = joblib.load(model_path)

result = 'result1'
set_names = ['train','val','test']
for set_name in set_names:

# 加载验证集
    validate_data = pd.read_csv('/data/split/'+ set_name + '_set.csv')
# 分离输入特征
    X_validate = validate_data.iloc[:, 1:35]

    # 使用SimpleImputer处理验证集中的缺失数据
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
    X_validate_imputed = imputer.fit_transform(X_validate)

    # 进行推理
    y_pred_proba = logreg.predict_proba(X_validate_imputed)[:, 1]  # 获取正类的概率值
    y_pred = (y_pred_proba > 0.5).astype(int)  # 转换为二分类预测结果
    y_pred_str = pd.Series(y_pred).map({0: 'Low Risk', 1: 'High Risk'})
    # 将预测结果添加到验证数据表格中
    validate_data['predicted_proba'] = y_pred_proba
    validate_data['predicted_label'] = y_pred

    # 保存预测结果到新的文件
    validate_data.to_csv('/class_result/'+ set_name + '_logic_predictions.csv', index=False)

    # 计算C-index
    y_time_validate = validate_data.iloc[:, 36]  # 生存时间
    c_index = concordance_index(y_time_validate, y_pred_proba)
    print(f'C-index: {c_index:.3f}')

    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(validate_data.iloc[:, 35], y_pred_proba)  # 35列是生存状态
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc:.3f}')

    # 绘制ROC曲线并保存到本地
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # 保存ROC曲线图像
    plt.savefig('/class_result/'+ set_name + 'roc_logic.png')
