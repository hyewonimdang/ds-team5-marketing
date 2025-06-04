import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

X = df[["Age_scaled", "Recency_scaled", "membership_Years_scaled", 
        "NumWebVisitsMonth_scaled", "Kidhome_scaled", "Teenhome_scaled", "Total_Purchases_scaled", 
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

# K-Fold 반복
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train KNN model (K=23)
    model = KNeighborsRegressor(n_neighbors=23)
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


# 결과 출력
print("<K-Nearest Neighbor Regression(K=23) Results>")
print("-R²-")
print("Train :", r2_train_list)
print("Test :", r2_list)
print("-MAE-")
print("Train :", mae_train_list)
print("Test :", mae_list)
print("-RMSE-")
print("Train :", [float(x) for x in rmse_train_list])
print("Test :", [float(x) for x in rmse_list])
print("-MSE-")
print("Train :", mse_train_list)
print("Test :", mse_list)
