from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read values from the form
        features = {
            "Temp": float(request.form["Temp"]),
            "Humidity": float(request.form["Humidity"]),
            "Cloud Cover": float(request.form["Cloud-Cover"]),
            "ANNUAL": float(request.form["ANNUAL"]),
            "Jan-Feb": float(request.form["Jan-Feb"]),
            "Mar-May": float(request.form["Mar-May"]),
            "Jun-Sep": float(request.form["Jun-Sep"]),
            "Oct-Dec": float(request.form["Oct-Dec"]),
            "avgjune": float(request.form["avgjune"]),
            "sub": float(request.form["sub"])
        }
        # Create DataFrame with correct column names
        data = pd.DataFrame([features])
        # Scale the input
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)

        if prediction[0] == 1:
            result = "⚠️ High Chance of Flood"
        else:
            result = "✅ Low Chance of Flood"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)