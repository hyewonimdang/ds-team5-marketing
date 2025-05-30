import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# 데이터 불러오기
df = pd.read_csv(r"C:\Users\User\Desktop\github\data science\ds-team5-marketing\data\cleaned_marketing_campaign.csv", sep=",")
df.to_csv(r"C:\Users\User\Desktop\github\data science\ds-team5-marketing\data\_marketing_campaign.csv", index=False)

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
# 독립변수, 종속변수 설정
X = df[["Age_scaled", "Recency_scaled", "membership_Years_scaled", 
        "Income_eq_width_Medium", "Income_eq_width_High", "Income_eq_freq_Medium", "Income_eq_freq_High", 
        "NumWebVisitsMonth_scaled", "Kidhome", "Teenhome", "Complain", "Total_Mnt_scaled",
         "Education_Graduation", "Education_Master", "Education_PhD",
         "Marital_Status_Married", "Marital_Status_Single", "Marital_Status_Together",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]]
y = df["Revenue_YN"]

# K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# 평가 지표 저장 리스트
acc_list_poly, prec_list_poly, rec_list_poly, f1_list_poly = [], [], [], []
acc_train_list_poly, prec_train_list_poly, rec_train_list_poly, f1_train_list_poly = [], [], [], []

confusion_matrices = []

# K-Fold 반복
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Polynomial Logistic Regression (2차 다항식)
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
    )
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # 성능 저장 (Train)
    acc_train_list_poly.append(accuracy_score(y_train, y_train_pred))
    prec_train_list_poly.append(precision_score(y_train, y_train_pred))
    rec_train_list_poly.append(recall_score(y_train, y_train_pred))
    f1_train_list_poly.append(f1_score(y_train, y_train_pred))

    # 성능 저장 (Test)
    acc_list_poly.append(accuracy_score(y_test, y_test_pred))
    prec_list_poly.append(precision_score(y_test, y_test_pred))
    rec_list_poly.append(recall_score(y_test, y_test_pred))
    f1_list_poly.append(f1_score(y_test, y_test_pred))

    confusion_matrices.append(confusion_matrix(y_test, y_test_pred))

# 결과 출력
print("<Polynomial Logistic Regression Results>")
print("-Accuracy-")
print("Train :", acc_train_list_poly)
print("Test :", acc_list_poly)
print("-Precision-")
print("Train :", prec_train_list_poly)
print("Test :", prec_list_poly)
print("-Recall-")
print("Train :", rec_train_list_poly)
print("Test :", rec_list_poly)
print("-F1-")
print("Train :", f1_train_list_poly)
print("Test :", f1_list_poly)

avg_confusion_matrices = np.mean(confusion_matrices, axis=0).astype(int)

# confusion matrix
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