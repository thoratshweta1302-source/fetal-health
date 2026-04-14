import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv('data/fetal_health.csv')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# EDA: Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='fetal_health', data=df)
plt.title('Distribution of Fetal Health')
plt.savefig('Visuals/fetal_health_distribution.png')
plt.close()

# EDA: Visualize baseline value distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['baseline value'], bins=30, kde=True)
plt.title('Distribution of Baseline Fetal Heart Rate')
plt.savefig('Visuals/baseline_value_distribution.png')
plt.close()

# EDA: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('Visuals/correlation_heatmap.png')
plt.close()

# Separate features and target
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Adjust y for XGBoost (it expects labels starting from 0)
y_adjusted = y - 1  # Convert 1,2,3 to 0,1,2

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_adjusted, test_size=0.2, random_state=42, stratify=y_adjusted
)

# Initialize models
# FIX: Removed deprecated 'multi_class' parameter (removed in scikit-learn 1.5+)
#      Removed deprecated 'use_label_encoder' parameter from XGBClassifier
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))

# Hyperparameter tuning for XGBoost (best model)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    param_grid, cv=5, scoring='f1_macro', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best F1-Score:", grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nTuned XGBoost Performance:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))

# Save the model and scaler using native XGBoost format to avoid version warnings
best_model.get_booster().save_model('fetalai_model.json')   # Native XGBoost format (no version warnings)
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved successfully.")
print("Model saved as: fetalai_model.json")
print("Scaler saved as: scaler.pkl")
