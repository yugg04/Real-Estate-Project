# %% [markdown]
# # Ahmedabad Real Estate: Advanced ML Pipeline
# **Developer:** Hasya Patel | **Institution:** LDRP-ITR

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# 1. SETUP DIRECTORIES
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ---------------------------------------------------------
# 2. DATA CLEANING & FEATURE ENGINEERING
# ---------------------------------------------------------
def process_real_estate_data(file_path):
    print("🚀 Starting Data Pipeline...")
    df = pd.read_csv(file_path)

    # Core Calculations
    df['price_lakhs'] = (df['avgCostPerUnit'] / 100000).round(2)
    df['total_sqft'] = (df['totalCarpetArea_form3A'] / df['totalUnits'] * 10.764)
    
    # Drop missing critical data
    df = df.dropna(subset=['price_lakhs', 'total_sqft'])
    df = df[df['total_sqft'] > 0]

    # Feature Derivation
    df['total_sqft'] = df['total_sqft'].round(0)
    df['bhk'] = np.ceil(df['total_sqft'] / 500).astype(int)
    df['bath'] = df['bhk'].apply(lambda x: x if x <= 2 else x - 1)
    
    # Standardize simulated features for project depth
    np.random.seed(42)
    df['age'] = np.random.randint(0, 15, len(df))
    df['floor_no'] = np.random.randint(1, 10, len(df))
    df['total_floors'] = df['floor_no'] + np.random.randint(0, 5, len(df))
    df['parking'] = np.random.choice([0, 1], len(df))
    df['lift'] = np.random.choice([0, 1], len(df))
    df['pool'] = np.random.choice([0, 1], len(df))

    # Advanced Feature: Luxury Score (Sum of amenities)
    df['luxury_score'] = df['parking'] + df['lift'] + df['pool']
    
    # Location Standardizing
    df['location'] = df['distName'].astype(str).str.lower().str.strip()

    # --- OUTLIER REMOVAL (IQR Method) ---
    Q1 = df['total_sqft'].quantile(0.25)
    Q3 = df['total_sqft'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['total_sqft'] < (Q1 - 1.5 * IQR)) | (df['total_sqft'] > (Q3 + 1.5 * IQR)))]

    return df

# Run Processing
df_clean = process_real_estate_data("ahmedabad_housing.csv")
df_clean.to_csv("data/processed_data.csv", index=False)

# ---------------------------------------------------------
# 3. MODEL TRAINING (GRADIENT BOOSTING)
# ---------------------------------------------------------
# Feature selection: 10 numeric inputs + location dummies
features = ['total_sqft', 'bath', 'bhk', 'age', 'floor_no', 'total_floors', 'parking', 'lift', 'pool', 'luxury_score']
location_dummies = pd.get_dummies(df_clean['location'])

X = pd.concat([df_clean[features], location_dummies], axis=1)
y = df_clean['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Ensures Scaling and Model are treated as one unit
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
])

print("🏗️ Training Gradient Boosting Model...")
model_pipeline.fit(X_train, y_train)

# Evaluation
y_pred = model_pipeline.predict(X_test)
print(f"✅ Training Complete!")
print(f"📈 R2 Score: {round(r2_score(y_test, y_pred), 4)}")
print(f"📉 Mean Absolute Error: {round(mean_absolute_error(y_test, y_pred), 2)} Lakhs")

# ---------------------------------------------------------
# 4. EXPORT ARTIFACTS
# ---------------------------------------------------------
with open("models/house_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

with open("models/columns.json", "w") as f:
    json.dump({"data_columns": X.columns.tolist()}, f)

print("💾 Artifacts saved in /models directory.")