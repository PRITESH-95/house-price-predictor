"""
House Price Prediction (Mumbai) - Simple Machine Learning Project
-----------------------------------------------------------------
Algorithm : Random Forest Regressor
Steps     : Load data -> Preprocess -> Train/Test split -> Train -> Evaluate -> Predict

Just keep the CSV file (or the unzipped file) in the same folder as this script
and run it:   python house_price_prediction.py
"""

import os
import glob
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =====================================================================
# STEP 1: LOAD THE DATASET
# =====================================================================
def find_and_load_dataset():
    """Looks for a CSV (or a zip containing a CSV) in the script's folder."""
    folder = os.path.dirname(os.path.abspath(__file__))

    # if only a zip file is present, extract it first
    if not glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True):
        for z in glob.glob(os.path.join(folder, "*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(folder)

    csv_files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("No CSV file found. Put the dataset in the same folder as this script.")

    path = csv_files[0]
    print("Loading dataset:", os.path.basename(path))
    return pd.read_csv(path)


print("=" * 60)
print("HOUSE PRICE PREDICTION")
print("=" * 60)

df = find_and_load_dataset()

# =====================================================================
# STEP 2: BASIC DATASET INFORMATION
# =====================================================================
print("\n--- DATASET INFORMATION ---")
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nStatistical summary of numerical columns:")
print(df.describe())

# =====================================================================
# STEP 3: PREPROCESSING
# =====================================================================
print("\n--- PREPROCESSING ---")

# 3.1 Find the target column (the house price) by looking at the column names
target = None
for col in df.columns:
    if "price" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
        target = col
        break
if target is None:
    raise ValueError("Could not find a price column in the dataset.")
print("Target column (what we predict):", target)

# 3.2 Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows. Rows left: {len(df)}")

# 3.3 Handle missing values
# numerical columns -> fill with median, categorical columns -> fill with most common value
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values handled (median for numbers, mode for text).")

# 3.4 The price is written in different units (Cr = Crore, L = Lakh).
# Mixing them would confuse the model, so convert everything into Lakhs.
# 1 Crore = 100 Lakhs
unit_col = None
for col in df.columns:
    if "unit" in col.lower():
        unit_col = col
        break

if unit_col is not None:
    is_crore = df[unit_col].astype(str).str.strip().str.lower().str.startswith("cr")
    df[target] = np.where(is_crore, df[target] * 100, df[target])
    df = df.drop(columns=[unit_col])  # unit column is no longer needed
    print(f"Converted '{target}' from Crore/Lakh into Lakhs using column '{unit_col}'.")

# 3.5 Drop ID-like columns (they do not help in predicting the price)
for col in list(df.columns):
    if col != target and ("id" == col.lower() or col.lower().endswith("_id") or col.lower() == "index"):
        df = df.drop(columns=[col])
        print("Dropped ID column:", col)

# 3.6 Drop columns that have too many different text values (like building names).
# One-hot encoding them would create thousands of columns, so we skip them.
for col in list(df.columns):
    if col != target and not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 300:
        print(f"Dropped '{col}' (too many unique values: {df[col].nunique()})")
        df = df.drop(columns=[col])

# 3.7 Remove unrealistic outliers using price per unit area (removes bottom 1% and top 1%)
area_col = None
for col in df.columns:
    if "area" in col.lower() or "sqft" in col.lower():
        area_col = col
        break

if area_col is not None:
    price_per_area = df[target] / df[area_col]
    low, high = price_per_area.quantile(0.01), price_per_area.quantile(0.99)
    before = len(df)
    df = df[(price_per_area >= low) & (price_per_area <= high)]
    print(f"Removed {before - len(df)} outlier rows (very unusual price per {area_col}).")

# 3.8 Separate features (X) and target (y)
X = df.drop(columns=[target])
y = df[target]

# 3.9 Identify numerical and categorical features
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numerical features  :", num_cols)
print("Categorical features:", cat_cols)

# 3.10 Encoding: categorical -> One-Hot Encoding
# Scaling is NOT needed because Random Forest is a tree-based model
# (it does not depend on the size of the numbers), so numerical columns pass as they are.
print("Encoding: One-Hot Encoding on categorical features.")
print("Scaling : Not required for Random Forest.")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",  # numerical columns are kept as they are
)

# =====================================================================
# STEP 4: TRAIN / TEST SPLIT
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {len(X_train)} | Testing rows: {len(X_test)}")

# =====================================================================
# STEP 5: MODEL TRAINING
# =====================================================================
print("\n--- MODEL ---")
print("Model used: Random Forest Regressor (100 trees)")
print("Why? It works well on mixed data, handles non-linear relationships,")
print("and does not need feature scaling.")

# Pipeline = preprocessing + model together,
# so the exact same preprocessing is automatically applied to new user input.
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])

print("\nTraining the model, please wait...")
model.fit(X_train, y_train)
print("Training completed.")

# =====================================================================
# STEP 6: EVALUATION
# =====================================================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION (on test data) ---")
print(f"MAE  (Mean Absolute Error)    : {mae:.2f} Lakhs")
print(f"MSE  (Mean Squared Error)     : {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} Lakhs")
print(f"R2 Score                      : {r2:.4f}")

# =====================================================================
# STEP 7: PREDICT PRICE FROM USER INPUT (taken from the terminal)
# =====================================================================
def format_price(lakhs):
    """Shows the price in Crore if it is 100 Lakhs or more."""
    if lakhs >= 100:
        return f"Rs. {lakhs / 100:.2f} Crore  ({lakhs:.2f} Lakhs)"
    return f"Rs. {lakhs:.2f} Lakhs"


def ask_number(col):
    """Keeps asking until the user types a valid positive number."""
    low, high = X[col].min(), X[col].max()
    while True:
        value = input(f"Enter {col} (dataset range {low:g} to {high:g}): ").strip()
        try:
            number = float(value)
            if number <= 0:
                print("  Value must be greater than 0.")
                continue
            return number
        except ValueError:
            print("  Please enter a valid number.")


def ask_category(col):
    """Asks the user to choose a category value seen in the dataset."""
    options = sorted(X[col].unique().tolist())

    # Few options -> show a numbered menu, user just types the number
    if len(options) <= 10:
        print(f"\nSelect {col}:")
        for i, option in enumerate(options, start=1):
            print(f"  {i}. {option}")
        while True:
            choice = input(f"Enter choice number (1-{len(options)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            print("  Invalid choice, try again.")

    # Many options (like region) -> user types the name, we help with suggestions
    lookup = {str(o).lower(): o for o in options}
    print(f"\n{col} has {len(options)} options. Examples: {X[col].value_counts().head(6).index.tolist()}")
    while True:
        value = input(f"Enter {col} name: ").strip().lower()
        if value in lookup:
            return lookup[value]
        # suggest names that contain the typed text
        matches = [o for o in options if value and value in str(o).lower()]
        if matches:
            print("  Not an exact match. Did you mean one of these?", matches[:8])
        else:
            print("  Not found in the dataset. Please try again.")


print("\n" + "=" * 60)
print("PREDICT A HOUSE PRICE")
print("=" * 60)
print("Please enter the details of the house when asked.")

while True:
    user_data = {}
    for col in X.columns:  # asks for exactly the columns the model was trained on
        if col in num_cols:
            user_data[col] = ask_number(col)
        else:
            user_data[col] = ask_category(col)

    # Convert the input into a DataFrame with the same columns as the training data
    user_df = pd.DataFrame([user_data])[X.columns]
    for col in num_cols:
        user_df[col] = user_df[col].astype(X[col].dtype)

    # The pipeline applies the same encoding automatically, then predicts
    predicted_price = model.predict(user_df)[0]

    print("\n" + "-" * 60)
    print("Details entered:")
    for col, value in user_data.items():
        print(f"  {col}: {value}")
    print("\nEstimated House Price:", format_price(predicted_price))
    print("-" * 60)

    again = input("\nPredict another house? (y/n): ").strip().lower()
    if again != "y":
        break

# =====================================================================
# STEP 8: GRAPH (shown at the end so it does not block the user input)
# =====================================================================
print("\nShowing the Actual vs Predicted graph (close the window to exit).")
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.3, s=8)
limit = max(y_test.max(), y_pred.max())
plt.plot([0, limit], [0, limit], color="red", label="Perfect prediction")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()
print("Thank you!")
