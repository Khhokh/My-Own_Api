from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
with open("disease_model (2).pkl", "rb") as f:
    model = pickle.load(f)

# Load the encoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load symptom columns
with open("X_columns.pkl", "rb") as f:
    X_columns = pickle.load(f)

# Create symptom index dictionary for lookup
symptom_index = {symptom: idx for idx, symptom in enumerate(X_columns)}
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')
    
# Function to predict disease
def predict_disease(symptoms):
    symptoms_list = symptoms.split(",")

    # Convert symptoms to numerical input
    input_data = np.zeros(len(X_columns))

    for symptom in symptoms_list:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1  # Mark symptom as present

    # Convert input_data into DataFrame
    input_df = pd.DataFrame([input_data], columns=X_columns)

    # Make prediction
    prediction = encoder.inverse_transform([model.predict(input_df)[0]])[0]

    return {"predicted_disease": prediction}

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')




# Flask API route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        symptoms = data.get("symptoms", "")

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        result = predict_disease(symptoms)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
