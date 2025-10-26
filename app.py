from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# تحميل الموديلات والإنكودر
with open("model_disease.pkl", "rb") as f:
    model_disease = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("target_encoder_disease.pkl", "rb") as f:
    target_encoder_disease = pickle.load(f)


@app.route('/')
def home():
    return jsonify({"message": "Disease Prediction API is running 🚀"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات كـ JSON
        data = request.get_json()

        chrom = data.get("Chrom")
        position = data.get("Position")
        ref = data.get("Ref")
        alt = data.get("Alt")
        clnsig = data.get("CLNSIG")

        # التحقق من القيم
        if None in [chrom, position, ref, alt, clnsig]:
            return jsonify({"error": "Missing one or more input fields"}), 400

        # تجهيز الداتا بنفس ترتيب التدريب
        X_input = np.array([[chrom, position, ref, alt, clnsig]])

        # تحويل النصوص إلى أرقام بالإنكودر
        for i, col in enumerate(['Chrom', 'Position', 'Ref', 'Alt', 'CLNSIG']):
            if col in label_encoders:
                le = label_encoders[col]
                X_input[:, i] = le.transform(X_input[:, i])

        X_input = X_input.astype(float)

        # التوقع من الموديل
        disease_pred = model_disease.predict(X_input)
        disease_name = target_encoder_disease.inverse_transform(disease_pred)[0]

        # حساب احتمالية الخطورة (risk probability)
        risk_prob = model_disease.predict_proba(X_input)[0].max()

        # تحديد مستوى الخطورة
        if risk_prob >= 0.8:
            risk_level = "High"
        elif risk_prob >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # إخراج النتيجة
        return jsonify({
            "Predicted_Disease": disease_name,
            "Risk_Prob": round(float(risk_prob), 3),
            "Risk_Level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
