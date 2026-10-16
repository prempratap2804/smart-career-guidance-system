# =============================================================
#  Smart Career Guidance System (v2 — no Career_Goal leakage)
#  File    : train_model.py
#  Task    : Train a Random Forest model on the new dataset and
#            save it as .pkl
#  Run     : python train_model.py
# =============================================================

import pandas as pd
import pickle
from sklearn.preprocessing   import LabelEncoder
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, classification_report


# -------------------------------------------------------------
# STEP 1 : Load the dataset
# -------------------------------------------------------------

df = pd.read_csv("smart_career_dataset.csv")

print("=" * 55)
print("  SMART CAREER GUIDANCE SYSTEM v2 — MODEL TRAINING")
print("=" * 55)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# -------------------------------------------------------------
# STEP 2 : Check for null values and duplicates
# -------------------------------------------------------------

null_count = df.isnull().sum().sum()
dup_count  = df.duplicated().sum()

print(f"\n[2] Null values  : {null_count}")
print(f"    Duplicates   : {dup_count}")

if dup_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"    Duplicates removed. Rows remaining: {len(df)}")
else:
    print(f"    No duplicates found. Rows: {len(df)}")


# -------------------------------------------------------------
# STEP 3 : Encode all categorical columns using LabelEncoder
#
#   Every input feature here is a single categorical value
#   (no comma-separated skill lists this time), so a plain
#   LabelEncoder per column is enough — no MultiLabelBinarizer
#   needed.
# -------------------------------------------------------------

print("\n[3] Encoding categorical columns with LabelEncoder...")

FEATURE_COLUMNS = [
    "Interest_Area", "Aptitude_Level", "Communication_Skills",
    "Programming_Skill", "Mathematical_Ability", "Problem_Solving_Skill",
    "Creativity_Level", "Work_Preference",
]

label_encoders = {}

for col in FEATURE_COLUMNS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"    [{col}] classes: {list(le.classes_)}")

# Encode the target column separately
le_target = LabelEncoder()
df["Recommended_Career"] = le_target.fit_transform(df["Recommended_Career"])
print(f"    [Target] classes: {list(le_target.classes_)}")


# -------------------------------------------------------------
# STEP 4 : Build feature matrix X and target vector y
# -------------------------------------------------------------

X = df[FEATURE_COLUMNS]
y = df["Recommended_Career"]

print(f"\n[4] Feature matrix shape: {X.shape}")
print(f"    Target column        : Recommended_Career")


# -------------------------------------------------------------
# STEP 5 : Split data into Training set and Testing set (80:20)
# -------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n[5] Train/Test Split (80:20)")
print(f"    Training samples: {len(X_train)}")
print(f"    Testing  samples: {len(X_test)}")


# -------------------------------------------------------------
# STEP 6 : Train the Random Forest Classifier
# -------------------------------------------------------------

print("\n[6] Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=200,     # 200 decision trees inside the forest
    max_depth=12,         # keeps the model generalizing instead of memorizing
    min_samples_leaf=3,   # avoids overly specific leaves, also shrinks file size
    random_state=42
)

model.fit(X_train, y_train)

print("    Training complete.")


# -------------------------------------------------------------
# STEP 7 : Evaluate the model and print results
# -------------------------------------------------------------

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy  = accuracy_score(y_test,  y_test_pred)

print(f"\n[7] Model Evaluation")
print("=" * 55)
print(f"    Training Accuracy : {train_accuracy * 100:.2f}%")
print(f"    Testing  Accuracy : {test_accuracy  * 100:.2f}%")

print("\n    Classification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=le_target.classes_
))


# -------------------------------------------------------------
# STEP 8 : Save the trained model as career_model.pkl
# -------------------------------------------------------------

with open("career_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[8]  Saved: career_model.pkl")


# -------------------------------------------------------------
# STEP 9 : Save all encoders in a single file encoders.pkl
# -------------------------------------------------------------

encoders = {
    "label_encoders"  : label_encoders,     # one LabelEncoder per input column
    "le_target"       : le_target,          # Recommended_Career decoder
    "feature_columns" : FEATURE_COLUMNS,    # exact column order for prediction
}

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("[9] Saved: encoders.pkl")

print("\n" + "=" * 55)
print("  Training complete! Files saved:")
print("    career_model.pkl")
print("    encoders.pkl")
print("=" * 55)
