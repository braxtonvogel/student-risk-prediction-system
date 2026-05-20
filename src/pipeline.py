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

# -------------------------
# 2. CLEAN + CREATE TARGET
# -------------------------
df = df[df["final_result"].notnull()]

df["risk"] = df["final_result"].apply(
    lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
)

# -------------------------
# 3. ENCODE CATEGORICAL FEATURES
# -------------------------
categorical_cols = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability"
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------
# 4. FEATURES / TARGET
# -------------------------
X = df_encoded.drop(columns=["final_result", "risk", "id_student"], errors="ignore")
y = df_encoded["risk"]

# Remove any NaNs
X = X.fillna(0)

# -------------------------
# 5. TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# 6. MODEL
# -------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 7. EVALUATION
# -------------------------
preds = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nModel training complete.")