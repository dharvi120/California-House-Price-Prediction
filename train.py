# Import Libraries
import shap

print("SHAP imported successfully!")
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
# ML tools
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("models", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
#loading dataset

data = fetch_california_housing()
# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target


#display first five rows
print(df.head())

X=df.drop("Price", axis = 1)
y=df['Price']
X_train, X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
    )
print("Training set size", X_train.shape)
print("Testing set size",X_test.shape)

#  Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

#evaluation metrics
# Predictions
y_pred = linear_model.predict(X_test)

# Evaluation Metrics
linear_mae = mean_absolute_error(y_test, y_pred)
linear_mse = mean_squared_error(y_test, y_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, y_pred)

print("\n===== Linear Regression =====")
print("MAE :", linear_mae)
print("MSE :", linear_mse)
print("RMSE:", linear_rmse)
print("R² :", linear_r2)

#feature scaling


scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Retrain Model on Scaled Data
linear_model_scaled=LinearRegression()
linear_model_scaled.fit(X_train_scaled,y_train)

y_pred_scaled=linear_model_scaled.predict(X_test_scaled)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred_scaled))
print("R2 Score:", r2_score(y_test, y_pred_scaled))

#Ridge Regression

y_pred_ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train).predict(X_test_scaled)

ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, y_pred_ridge)

print("\n===== Ridge Regression =====")
print("MAE :", ridge_mae)
print("MSE :", ridge_mse)
print("RMSE:", ridge_rmse)
print("R² :", ridge_r2)
#Model Interpretation
coefficients = pd.DataFrame(
    linear_model_scaled.coef_,
    X.columns,
    columns=["Coefficient"]
)


# Random Forest Regression

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train Model
random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)

# Step 4: Calculate evaluation metrics
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
# Step 5: Print the results
print("\n========== Random Forest Results ==========")
print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)

# Step 6: Compare with previous models
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        ridge_mae,
        rf_mae
    ],
    "MSE": [
        linear_mse,
        ridge_mse,
        rf_mse
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
        rf_rmse
    ],
    "R2 Score": [
        linear_r2,
        ridge_r2,
        rf_r2
    ]
})

print("\n========== Model Comparison ==========")   
print(results)


# XGBoost Regression


# Step 1: Create the XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
# Step 2: Train the model
xgb_model.fit(X_train, y_train)
# Step 3: Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Step 4: Calculate evaluation metrics

# Step 4: Calculate evaluation metrics
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, y_pred_xgb)

# Step 5: Print the results
# Step 5: Print the results
print("\n===== XGBoost Regression =====")
print("MAE :", xgb_mae)
print("MSE :", xgb_mse)
print("RMSE:", xgb_rmse)
print("R² :", xgb_r2)

# Step 6: Update the comparison table
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        linear_mae,
        ridge_mae,
        rf_mae,
        xgb_mae
    ],
    "MSE": [
        linear_mse,
        ridge_mse,
        rf_mse,
        xgb_mse
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
        rf_rmse,
        xgb_rmse
    ],
    "R2 Score": [
        linear_r2,
        ridge_r2,
        rf_r2,
        xgb_r2
    ]
})

print("\n========== Model Comparison ==========")
print(results)

# Random Forest Feature Importance


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== Feature Importance =====")
print(feature_importance)

# Plot Feature Importance

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()

plt.show()


# Create SHAP Explainer
# ============================
# SHAP Explainability
# ============================

# Create SHAP Explainer
# plt.show()

# SHAP code starts here
print(shap.__version__)
plt.tight_layout()

plt.savefig(
    "screenshots/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
# ============================
# SHAP Explainability
# ============================

print("\n========== SHAP Explainability ==========")

# Take a small sample for faster explanation
X_sample = X_test.iloc[:100]

print("Creating SHAP Explainer...")

explainer = shap.Explainer(xgb_model)

print("Generating SHAP Values...")

shap_values = explainer(X_sample)

print("Done!")

print("SHAP Values Shape:")
print(shap_values.values.shape)

#SHAP Summary Plot

print("Creating SHAP Summary Plot...")

plt.figure(figsize=(10,7))

shap.plots.beeswarm(
    shap_values,
    show=False
)

plt.tight_layout()

plt.savefig(
    "screenshots/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================
# SHAP Waterfall Plot
# ============================

print("\nCreating Waterfall Plot...")

shap.plots.waterfall(shap_values[0])

plt.show()


plt.figure(figsize=(10,7))

shap.plots.beeswarm(
    shap_values,
    show=False
)

plt.tight_layout()

plt.savefig(
    "screenshots/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


results.to_csv(
    "models/model_metrics.csv",
    index=False
)

print("✅ Metrics saved.")


sample = X.iloc[[0]]

prediction = xgb_model.predict(sample)

print("\nSample Prediction")
print(prediction)
# ============================
# Save Trained Model
# ============================

os.makedirs("models", exist_ok=True)

joblib.dump(xgb_model, "models/xgboost_model.pkl")

print("✅ XGBoost model saved successfully!")
