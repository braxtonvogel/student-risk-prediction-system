import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# 1. LOAD DATA
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "studentInfo.csv")

df = pd.read_csv(file_path)

print("Columns in dataset:")
print(df.columns)

# -------------------------
# 2. CLEAN FEATURES (AUTO-SELECT SAFE NUMERIC ONES)
# -------------------------
# This avoids the KeyError problem you just had
X = df.select_dtypes(include=['number']).dropna()

# If your dataset has a target column, try to auto-detect it
possible_targets = [col for col in df.columns if "result" in col.lower() or "risk" in col.lower()]

if len(possible_targets) > 0:
    target_col = possible_targets[0]
else:
    target_col = df.columns[-1]  # fallback (last column)

y = df[target_col]

# Align X and y safely
X = X.loc[y.index]

# -------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# 4. MODEL
# -------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 5. EVALUATION
# -------------------------
preds = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nTarget column used:", target_col)

print("\nModel training complete.")