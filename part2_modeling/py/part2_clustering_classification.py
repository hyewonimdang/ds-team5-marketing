import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

df = pd.read_csv(r"C:\Users\User\Desktop\github\data science\ds-team5-marketing\data\cleaned_marketing_campaign.csv", sep=",")

# X 선정을 위한 coefficient 확인
# 전체 feature 지정 (타겟 변수와 직접적인 연관 있는 feature 제외)
candidate_features = df.columns.drop(["Revenue_YN", "Total_Purchases_scaled"]) 
X_candidates = df[candidate_features]
y = df["Revenue_YN"]

# 모델 학습
model = LogisticRegression(class_weight='balanced',solver='liblinear')
model.fit(X_candidates, y)

# coefficient 확인
coef_df = pd.DataFrame({
    'Feature': candidate_features,
    'Coefficient': model.coef_[0]
})

# 절대값 기준 내림차순 정렬
coef_df['abs_coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

print(coef_df[['Feature', 'Coefficient']])

X = df[["Age_scaled", "Recency_scaled", "membership_Years_scaled", "Income_scaled", 
        "NumWebVisitsMonth_scaled", "Kidhome_scaled", "Teenhome_scaled", "Complain", "Total_Mnt_scaled",
         "Education_Basic", "Education_Graduation", "Education_Master", "Education_PhD",
         "Marital_Status_Alone", "Marital_Status_Divorced", "Marital_Status_Widow", "Marital_Status_YOLO",
         "Marital_Status_Married", "Marital_Status_Single", "Marital_Status_Together",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]]

y = df["Revenue_YN"]

# KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

acc_list, prec_list, rec_list, f1_list = [], [], [], []
acc_train_list, prec_train_list, rec_train_list, f1_train_list = [], [], [], []

confusion_matrices = []

# K-Fold 반복
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train KNN model (K=13)
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    acc_train_list.append(accuracy_score(y_train, y_train_pred))
    prec_train_list.append(precision_score(y_train, y_train_pred))
    rec_train_list.append(recall_score(y_train, y_train_pred))
    f1_train_list.append(f1_score(y_train, y_train_pred))

    acc_list.append(accuracy_score(y_test, y_test_pred))
    prec_list.append(precision_score(y_test, y_test_pred))
    rec_list.append(recall_score(y_test, y_test_pred))
    f1_list.append(f1_score(y_test, y_test_pred))

    confusion_matrices.append(confusion_matrix(y_test, y_test_pred))

# 결과 출력
print("<K-Nearest Neighbor Classification(K=13) Results>")
print("-Accuracy-")
print("Train :", acc_train_list)
print("Test :", acc_list)
print("-Precision-")
print("Train :", prec_train_list)
print("Test :", prec_list)
print("-Recall-")
print("Train :", rec_train_list)
print("Test :", rec_list)
print("-F1-")
print("Train :", f1_train_list)
print("Test :", f1_list)

avg_confusion_matrices = np.mean(confusion_matrices, axis=0).astype(int)
sns.heatmap(avg_confusion_matrices, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("Average Confusion Matrix (5-Fold)")

# ROC Curve 및 AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
auc_score = roc_auc_score(y_test, y_test_proba)

# 그래프 출력
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color="darkorange", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # 랜덤 모델 기준선
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Polynomial Logistic Regression (Last Fold)")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()