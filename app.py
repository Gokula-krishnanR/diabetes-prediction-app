from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load both scaler and model
scaler, model = joblib.load("best_model.joblib")

# 8 Features (matching training)
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data from form
        data = [float(request.form[f]) for f in FEATURES]
        X = np.array(data).reshape(1, -1)

        # Scale and predict
        X_scaled = scaler.transform(X)
        prediction = int(model.predict(X_scaled)[0])
        prob = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else 0.5

        # Prepare message
        msg = "🩸 Diabetic" if prediction else "✅ Not Diabetic"
        return render_template("index.html", result=msg, probability=f"{prob*100:.1f}%")

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
