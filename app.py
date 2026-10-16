#===================================================
#  Smart Career Guidance System v2
#  File : app.py
#  Run  : python app.py
# =============================================================

import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


# -------------------------------------------------------------
# Load the trained model and encoders at startup
# -------------------------------------------------------------

with open("career_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

label_encoders  = encoders["label_encoders"]   # one LabelEncoder per input column
le_target       = encoders["le_target"]        # decodes prediction number -> career name
feature_columns = encoders["feature_columns"]  # exact column order used during training

print("Model and encoders loaded successfully.")


# -------------------------------------------------------------
# Allowed values for input validation
# -------------------------------------------------------------

ALLOWED_VALUES = {
    "interest_area":         ["AI/ML", "Data", "Web Development", "Software Development",
                               "Cybersecurity", "Networking", "Cloud Computing", "Design",
                               "Business", "Marketing"],
    "aptitude_level":        ["Low", "Medium", "High"],
    "communication_skills":  ["Poor", "Average", "Good", "Excellent"],
    "programming_skill":     ["Beginner", "Intermediate", "Advanced"],
    "mathematical_ability":  ["Low", "Medium", "High"],
    "problem_solving_skill": ["Low", "Medium", "High"],
    "creativity_level":      ["Low", "Medium", "High"],
    "work_preference":       ["Individual", "Team", "Both"],
}

# Maps the JSON field name (snake_case, from the frontend) to the
# exact column name the model/encoders were trained on
FIELD_TO_COLUMN = {
    "interest_area":         "Interest_Area",
    "aptitude_level":        "Aptitude_Level",
    "communication_skills":  "Communication_Skills",
    "programming_skill":     "Programming_Skill",
    "mathematical_ability":  "Mathematical_Ability",
    "problem_solving_skill": "Problem_Solving_Skill",
    "creativity_level":      "Creativity_Level",
    "work_preference":       "Work_Preference",
}


# -------------------------------------------------------------
# ROUTE 1 : GET /
# -------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------------------
# ROUTE 2 : POST /predict
# -------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided."}), 400

    # Step 1: Check all required fields are present
    required_fields = list(ALLOWED_VALUES.keys())
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: '{field}'"}), 400

    # Step 2: Validate every field against its allowed values
    for field in required_fields:
        value = data[field]
        if value not in ALLOWED_VALUES[field]:
            return jsonify({
                "error": f"Invalid value for '{field}': '{value}'. Allowed: {ALLOWED_VALUES[field]}"
            }), 422

    # Step 3: Encode each field using its saved LabelEncoder
    try:
        encoded = {}
        for field, column in FIELD_TO_COLUMN.items():
            encoded[column] = label_encoders[column].transform([data[field]])[0]
    except Exception as e:
        return jsonify({"error": f"Encoding error: {str(e)}"}), 500

    # Step 4: Build a single-row DataFrame in the exact training column order
    row = pd.DataFrame([encoded])
    row = row.reindex(columns=feature_columns)

    # Step 5: Predict
    prediction_encoded = model.predict(row)[0]
    career = le_target.inverse_transform([prediction_encoded])[0]

    # Step 6: Also return the top-3 most likely careers with confidence,
    # since Random Forest gives us class probabilities for free
    proba = model.predict_proba(row)[0]
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [
        {"career": le_target.inverse_transform([i])[0], "confidence": round(float(proba[i]) * 100, 1)}
        for i in top3_idx
    ]

    return jsonify({"career": career, "top3": top3})


# -------------------------------------------------------------
# Run the Flask development server
# -------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
