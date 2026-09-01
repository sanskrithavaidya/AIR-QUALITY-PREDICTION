import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNG files
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Loading dataset...")

data = pd.read_csv("AirQuality (1).csv", sep=';', decimal=',', encoding='latin1')

print("Dataset loaded")

# remove completely empty column
data = data.dropna(axis=1, how='all')

# replace missing value indicator
data = data.replace(-200, pd.NA)

print("Cleaning dataset")

# convert only required columns to numeric
FEATURE_NAMES = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
                  'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
columns = FEATURE_NAMES + ['CO(GT)']

for col in columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# drop rows where target is missing
data = data.dropna(subset=['CO(GT)'])

print("Remaining rows:", len(data))

# features
X = data[FEATURE_NAMES]

# target
y = data['CO(GT)']

print("Splitting dataset")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate the model ----
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"R2 score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

print("Saving model")
pickle.dump(model, open("model.pkl", "wb"))

# save the feature name order so app.py always matches training
pickle.dump(FEATURE_NAMES, open("feature_names.pkl", "wb"))

print("✅ Model saved successfully!")

# ---------------------------------------------------------------
# Explainable AI: SHAP global feature importance
# ---------------------------------------------------------------
print("Computing SHAP values for global explainability...")

# TreeExplainer is built specifically for tree-based models like RandomForest
explainer = shap.TreeExplainer(model)

# use a sample of the test set so this stays fast on larger datasets
sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
shap_values = explainer.shap_values(sample)

# save the explainer too, so app.py doesn't need to rebuild it every run
pickle.dump(explainer, open("explainer.pkl", "wb"))

# 1) Global bar chart: average impact of each feature on the prediction
plt.figure()
shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
plt.title("Global Feature Importance (mean |SHAP value|)")
plt.tight_layout()
plt.savefig("static/shap_summary_bar.png", dpi=150)
plt.close()

# 2) Beeswarm plot: shows direction + spread of each feature's effect
plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("static/shap_summary_beeswarm.png", dpi=150)
plt.close()

print("✅ SHAP global explanation plots saved to static/")
