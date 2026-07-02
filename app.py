import os
import time
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Initialize timer, constants, and local directories
START = time.time()
DATA_PATH = "insurance_retirement.csv"
TARGET_COL = "Expected Corpus"
os.makedirs("models", exist_ok=True)

# Load and clean data
df = pd.read_csv(DATA_PATH).drop_duplicates()

# Separate features from target
X = df.drop(columns=TARGET_COL)
y = df[TARGET_COL].astype(float)

# Separate columns dynamically by data type
num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns

prep = ColumnTransformer([
    ("num", Pipeline([
        ("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("i", SimpleImputer(strategy="most_frequent")),
        ("o", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

# Split into training and testing sets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and apply data transformations
prep.fit(Xtr)
Xtrp = prep.transform(Xtr)
Xtep = prep.transform(Xte)

# Track resulting feature names
feat = prep.get_feature_names_out()

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=180, max_depth=12, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=220, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42)
}

res = {}
trained = {}

# Train and evaluate each model
for n, m in models.items():
    m.fit(Xtrp, ytr)
    a = m.predict(Xtrp)
    b = m.predict(Xtep)
    trained[n] = m
    res[n] = {
        "Train_R2": r2_score(ytr, a),
        "Test_R2": r2_score(yte, b),
        "Train_RMSE": np.sqrt(mean_squared_error(ytr, a)),
        "Test_RMSE": np.sqrt(mean_squared_error(yte, b)),
        "Train_MAE": mean_absolute_error(ytr, a),
        "Test_MAE": mean_absolute_error(yte, b)
    }

# Rank models based on Test R² performance
results = pd.DataFrame(res).T.sort_values("Test_R2", ascending=False)
print(results.round(4))

best = results.index[0]

# Construct final end-to-end pipeline
pipe = Pipeline([("prep", prep), ("model", trained[best])])

# Validate across the complete dataset
cv = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"CV Mean R2: {cv.mean()}, CV Std Dev: {cv.std()}")

# Save full pipeline artifact
joblib.dump(pipe, "models/Best_Model.joblib")


# Downsample for faster SHAP calculations
idx = np.random.default_rng(42).choice(Xtrp.shape[0], min(10000, Xtrp.shape[0]), replace=False)
Xs = Xtrp[idx]

# Dynamically select explainer type based on the winning model
if best != "LinearRegression":
    vals = shap.TreeExplainer(trained[best]).shap_values(Xs)
else:
    vals = shap.Explainer(trained[best], Xs)(Xs)

# Render summary plot and display pipeline runtime
shap.summary_plot(vals, Xs, feature_names=feat, show=False)
plt.tight_layout()
plt.show()

print(f"Total Execution Time: {time.time() - START} seconds")

# Optimized salary prediction pipeline with Universal Explainability Extensions
import os, time, joblib, shap, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# --- Existing Pipeline Setup ---
START = time.time(); DATA_PATH = "insurance_retirement.csv"; TARGET_COL = "Expected Corpus"; os.makedirs("models", exist_ok=True)
df = pd.read_csv(DATA_PATH).drop_duplicates(); X = df.drop(columns=TARGET_COL); y = df[TARGET_COL].astype(float)
num = X.select_dtypes(include="number").columns; cat = X.select_dtypes(exclude="number").columns
prep = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num), ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
prep.fit(Xtr); Xtrp = prep.transform(Xtr); Xtep = prep.transform(Xte); feat = prep.get_feature_names_out()
models = {"LinearRegression": LinearRegression(), "RandomForest": RandomForestRegressor(n_estimators=180, max_depth=12, n_jobs=-1, random_state=42), "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=220, max_depth=8, learning_rate=.08, subsample=.8, colsample_bytree=.8, tree_method="hist", n_jobs=-1, random_state=42)}
res = {}; trained = {}
for n, m in models.items():
    m.fit(Xtrp, ytr); a = m.predict(Xtrp); b = m.predict(Xtep); trained[n] = m
    res[n] = {"Train_R2": r2_score(ytr, a), "Test_R2": r2_score(yte, b), "Train_RMSE": np.sqrt(mean_squared_error(ytr, a)), "Test_RMSE": np.sqrt(mean_squared_error(yte, b)), "Train_MAE": mean_absolute_error(ytr, a), "Test_MAE": mean_absolute_error(yte, b)}
results = pd.DataFrame(res).T.sort_values("Test_R2", ascending=False); print(results.round(4)); best = results.index[0]; pipe = Pipeline([("prep", prep), ("model", trained[best])]); cv = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1); print(cv.mean(), cv.std()); joblib.dump(pipe, "models/Best_Model.joblib")
idx = np.random.default_rng(42).choice(Xtrp.shape[0], min(2000, Xtrp.shape[0]), replace=False); Xs = Xtrp[idx]

# Dynamically calculate SHAP values based on best model architecture
if best != "LinearRegression":
    explainer = shap.TreeExplainer(trained[best])
    shap_matrix = explainer.shap_values(Xs)
    # Handle SHAP output structure variations across different tree package versions cleanly
    base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else:
    explainer = shap.Explainer(trained[best], Xs)
    shap_obj = explainer(Xs)
    shap_matrix = shap_obj.values
    base_val = shap_obj.base_values[0] if isinstance(shap_obj.base_values, (list, np.ndarray)) else shap_obj.base_values

shap.summary_plot(shap_matrix, Xs, feature_names=feat, show=False); plt.tight_layout(); plt.show(); print(f"Baseline Pipeline Time: {time.time()-START}s")

# =====================================================================
# ADDITION 1: NLP-Style Natural Language Explainer (Universal & Dynamic)
# =====================================================================
print("\n" + "="*50 + "\nNLP-STYLE INDIVIDUAL PREDICTION STORY\n" + "="*50)

# 1. Pick a random test profile from our processed sample matrix
sample_idx = np.random.default_rng().choice(Xs.shape[0])
individual_features = Xs[sample_idx]
individual_shap = shap_matrix[sample_idx]

# 2. Extract the actual prediction mathematically matching the SHAP matrix
individual_pred = base_val + np.sum(individual_shap)

print(f"Starting Baseline Value (Average Dataset Prediction): {base_val:,.2f}")
print(f"Final Model Prediction for this Individual: {individual_pred:,.2f}\n")
print(f"--- How the Model Arrived at this Decision (Top Drivers) ---")

# 3. Zip features and evaluate magnitude directions dynamically without static text assumptions
drivers = pd.DataFrame({
    'Feature': feat,
    'SHAP_Value': individual_shap,
    'Absolute_Impact': np.abs(individual_shap)
}).sort_values(by='Absolute_Impact', ascending=False)

# 4. Generate dynamic plain-English narratives for the top 5 most impactful features
for _, row in drivers.head(5).iterrows():
    direction = "HIGHER" if row['SHAP_Value'] > 0 else "LOWER"
    action = "pushed UP" if row['SHAP_Value'] > 0 else "pulled DOWN"
    print(f"• The feature '{row['Feature']}' {action} the prediction by {abs(row['SHAP_Value']):,.2f}, leading to a {direction} final estimate.")

# ADDITION 2: Global Feature Importance Extraction (Universal & Dynamic)
# =====================================================================
print("\n" + "="*50 + "\nGLOBAL FEATURE IMPORTANCE SUMMARY\n" + "="*50)

# Mathematically aggregate the absolute SHAP matrices across all evaluated samples
global_impacts = np.mean(np.abs(shap_matrix), axis=0)

# Map back to variable column names dynamically
feature_importance_df = pd.DataFrame({
    'Feature_Name': feat,
    'Global_Impact_Score': global_impacts
}).sort_values(by='Global_Impact_Score', ascending=False).reset_index(drop=True)

# Display the ranked overall impact dataframe
print(feature_importance_df.head(10).round(4))



