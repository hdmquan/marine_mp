# %% Setup
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import PATH
import holoviews as hv

# %% Load and prepare data
df = pd.read_csv(PATH.PROCESSED / "microplastics_modis_combined.csv")

# Select features and target
features = [
    # 'latitude', 'longitude',
    "modis_sur_refl_b01_1",
    "modis_sur_refl_b02_1",
    "modis_sur_refl_b03_1",
    "modis_sur_refl_b04_1",
    "modis_sur_refl_b05_1",
]
X = df[features]
y = df["mp_concentration"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Train LightGBM model
model = lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# %% Evaluate model
y_pred = model.predict(X_test)

# Calculate metrics
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# %% Feature importance
importance = pd.DataFrame(
    {"feature": features, "importance": model.feature_importances_}
)
print("\nFeature Importance:")
print(importance.sort_values("importance", ascending=False))

# %% Plot predictions vs actual
hv.extension('bokeh')

# Create scatter plot
scatter = hv.Scatter(
    data=pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    }),
    kdims=['Actual'],
    vdims=['Predicted']
)

# Add 1:1 line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
line = hv.Curve([(min_val, min_val), (max_val, max_val)])

# Customize plot
plot = (scatter * line).opts(
    width=500,
    height=500,
    aspect='equal',  # This ensures 1:1 aspect ratio
    xlabel='Actual MP Concentration',
    ylabel='Predicted MP Concentration',
    title='Predicted vs Actual MP Concentration',
    tools=['hover'],
)

plot

# %%
