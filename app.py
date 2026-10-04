from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import traceback

app = Flask(__name__)

# Load saved model and preprocessor
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "netflix_model.pkl"

saved_pipeline = joblib.load(MODEL_PATH)

model = saved_pipeline["model"]
preprocessor = saved_pipeline["preprocessor"]

# Get exact columns used when the preprocessor was trained
expected_columns = preprocessor.feature_names_in_.tolist()

print("Model expected columns:", expected_columns)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        release_year = int(request.form.get("release_year", "").strip())
        date_added = request.form.get("date_added", "").strip()
        rating = request.form.get("rating", "").strip()
        country = request.form.get("country", "").strip()
        listed_in = request.form.get("listed_in", "").strip()
        description = request.form.get("description", "").strip()

        # Convert date
        added_date = datetime.strptime(date_added, "%Y-%m-%d")

        added_year = added_date.year
        added_month = added_date.month
        added_month_name = added_date.strftime("%B")

        # Feature engineering
        country_count = len([
            item for item in country.split(",") if item.strip()
        ])

        genre_count = len([
            item for item in listed_in.split(",") if item.strip()
        ])

        description_length = len(description)

        # Prepare all available input features
        input_values = {
            "release_year": release_year,
            "added_year": added_year,
            "added_month": added_month,
            "added_month_name": added_month_name,
            "country_count": country_count,
            "genre_count": genre_count,
            "description_length": description_length,
            "rating": rating,
            "country": country,
            "listed_in": listed_in
        }

        # Create DataFrame
        input_data = pd.DataFrame([input_values])

        print("Input columns:", input_data.columns.tolist())

        # Add any columns expected by the saved preprocessor
        # that are not present in the input data
        input_data = input_data.reindex(columns=expected_columns)

        print("Final input columns:", input_data.columns.tolist())
        print(
            "Missing columns:",
            set(expected_columns) - set(input_data.columns)
        )

        # Transform input using the saved preprocessor
        input_processed = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_processed)[0]

        print("Prediction:", prediction)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as error:
        print("Prediction Error:", repr(error))
        traceback.print_exc()

        return render_template(
            "index.html",
            error=f"Prediction failed: {error}"
        )


if __name__ == "__main__":
    app.run(debug=False)