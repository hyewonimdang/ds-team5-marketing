import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 데이터 읽기
df = pd.read_csv(r"C:\Users\User\Desktop\github\data science\ds-team5-marketing\data\cleaned_marketing_campaign.csv", sep=",")

# X 선정을 위한 coefficient 확인
candidate_features = df.columns.drop(["Total_Mnt", "Total_Mnt_scaled"]) 
X_candidates = df[candidate_features]
y = df["Total_Mnt"]

# Lasso 모델 학습 및 coefficient 확인
lasso_for_coef = Lasso(alpha=0.1)
lasso_for_coef.fit(X_candidates, y)

coef_df = pd.DataFrame({
    'Feature': candidate_features,
    'Coefficient': lasso_for_coef.coef_
})

coef_df['abs_coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False)
print(coef_df[['Feature', 'Coefficient']])

# 독립변수, 종속변수 설정
# 앞서 확인한 feature들의 Coefficient를 기준으로 선정(scaled 되지 않은 값 제외)
X = df[["Age_scaled", "Recency_scaled", "membership_Years_scaled", 
        "NumWebVisitsMonth_scaled", "Kidhome", "Teenhome", "Total_Purchases_scaled", 
        "Income_eq_width_Medium", "Income_eq_width_High", "Income_eq_freq_Medium", "Income_eq_freq_High",
        "Education_Basic", "Education_Graduation", "Education_Master", "Education_PhD",
        "Marital_Status_Divorced", "Marital_Status_Widow", "Marital_Status_YOLO",
        "Marital_Status_Single", "Marital_Status_Together",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]]

y = df["Total_Mnt"]

# KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 결과 저장 리스트
mse_list, rmse_list, r2_list, mae_list = [], [], [], []
mse_train_list, rmse_train_list, r2_train_list, mae_train_list = [], [], [], []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 모델 생성 및 학습
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)

    mae_train_list.append(mae_train)
    mse_train_list.append(mse_train)
    rmse_train_list.append(rmse_train)
    r2_train_list.append(r2_train)

    mae_list.append(mae_test)
    mse_list.append(mse_test)
    rmse_list.append(rmse_test)
    r2_list.append(r2_test)

# feature improtance(마지막 K-Fold 기준)
feature_names = X.columns
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0]
})
coef_df['abs_coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

print("Top 10 Logistic Regression Coefficients (by absolute value):")
print(coef_df[['Feature', 'Coefficient']])


# 결과 출력
print("<Lasso Regression>")
print("-R²-")
print("Train :", r2_train_list)
print("Test :", r2_list)
print("-MAE-")
print("Train :", mae_train_list)
print("Test :", mae_list)
print("-RMSE-")
print("Train :",[float(x) for x in rmse_train_list])
print("Test :", [float(x) for x in rmse_list])
print("-MSE-")
print("Train :", mse_train_list)
print("Test :", mse_list)