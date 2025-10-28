from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load trained models and encoders
with open("model_disease.pkl", "rb") as f:
    model_disease = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("target_encoder_disease.pkl", "rb") as f:
    target_encoder_disease = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Disease Prediction API is running successfully!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        chrom = data.get("Chrom")
        position = data.get("Position")
        ref = data.get("Ref")
        alt = data.get("Alt")
        clnsig = data.get("CLNSIG")

        # ✅ Validate input
        if None in [chrom, position, ref, alt, clnsig]:
            return jsonify({"error": "Missing one or more input fields"}), 400

        X_input = np.array([[chrom, position, ref, alt, clnsig]])

        # ✅ Apply encoders if column exists
        for i, col in enumerate(["Chrom", "Position", "Ref", "Alt", "CLNSIG"]):
            if col in label_encoders:
                encoder = label_encoders[col]
                try:
                    X_input[:, i] = encoder.transform(X_input[:, i])
                except:
                    return jsonify({"error": f"Invalid value for {col}"}), 400

        X_input = X_input.astype(float)

        # ✅ Model prediction
        prediction = model_disease.predict(X_input)
        disease_name = target_encoder_disease.inverse_transform(prediction)[0]

        # ✅ Probability
        prob = model_disease.predict_proba(X_input)[0].max()
        risk_prob = float(prob)

        # ✅ Risk level
        if risk_prob >= 0.8:
            risk_level = "High"
        elif risk_prob >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # ✅ Response JSON
        return jsonify({
            "Predicted_Disease": disease_name,
            "Risk_Prob": round(risk_prob, 3),
            "Risk_Level": risk_level
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
