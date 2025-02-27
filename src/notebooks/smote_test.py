# %% Setup
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import PATH
import holoviews as hv
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import random
from scipy.interpolate import interp1d

# Add these at the top after imports
np.random.seed(42)
random.seed(42)

# %% Load and prepare data
df = pd.read_csv(PATH.PROCESSED / "microplastics_modis_combined.csv")

# Select features and target first
features = [
    # 'latitude', 'lbongitude',
    "modis_sur_refl_b01_1",
    "modis_sur_refl_b02_1",
    "modis_sur_refl_b03_1",
    "modis_sur_refl_b04_1",
    "modis_sur_refl_b05_1",
    "modis_sur_refl_b07_1",
]

# Create a clean subset with only our features and target
X = df[features].copy()
# Mean impute NaN values only for our selected features
X = X.fillna(X.mean())

y = pd.qcut(df["mp_concentration"], q=3, labels=["Low", "Medium", "High"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# %% Data Augmentation
def add_gaussian_noise(X, noise_factor=0.05):
    """Add random Gaussian noise to features"""
    np.random.seed(42)  # Add seed
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

def interpolate_points(X, num_points=1):
    """Create new samples by interpolating between existing points"""
    random.seed(42)  # Add seed
    new_samples = []
    for _ in range(num_points):
        idx1, idx2 = random.sample(range(len(X)), 2)
        alpha = random.random()
        interpolated = X.iloc[idx1] * alpha + X.iloc[idx2] * (1 - alpha)
        new_samples.append(interpolated)
    return pd.DataFrame(new_samples, columns=X.columns)

# Create augmented training data using multiple techniques
print("Applying data augmentation techniques...")

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train[features], y_train)

X_noise = pd.DataFrame(
    add_gaussian_noise(X_train.values),
    columns=X_train.columns
)
y_noise = y_train.copy()

X_interp = interpolate_points(X_train, num_points=len(X_train) // 2)
y_interp = y_train.sample(n=len(X_interp), replace=True, random_state=42).reset_index(drop=True)

# Combine all augmented datasets
X_train_balanced = pd.concat([
    X_smote,
    X_noise,
    X_interp
], axis=0)

y_train_balanced = pd.concat([
    pd.Series(y_smote),
    y_noise,
    y_interp
], axis=0)

# Print final dataset sizes
print("\nAugmented dataset sizes:")
print(f"Original training set: {len(X_train)}")
print(f"Augmented training set: {len(X_train_balanced)}")
print("\nClass distribution in augmented dataset:")
print(pd.Series(y_train_balanced).value_counts())

# %% Train LightGBM model
model = lgb.LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
model.fit(X_train_balanced, y_train_balanced)

# %% Evaluate model
y_pred = model.predict(X_test)

# Print classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# %% Feature importance
importance = pd.DataFrame(
    {"feature": features, "importance": model.feature_importances_}
)
print("\nFeature Importance:")
print(importance.sort_values("importance", ascending=False))

# %% Plot confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, 
            fmt='d',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

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

# Customize plot
plot = scatter.opts(
    width=500,
    height=500,
    xlabel='Actual MP Concentration',
    ylabel='Predicted MP Concentration',
    title='Predicted vs Actual MP Concentration',
    tools=['hover'],
    size=8,  # Make points bigger
    jitter=0.2  # Add some jitter to see overlapping points better
)

plot

# %%
model.booster_.save_model(PATH.WEIGHTS / "lgbm_model.txt")

# %%
