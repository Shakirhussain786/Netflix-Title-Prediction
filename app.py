# Import required libraries
from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

# Create Flask application
app = Flask(__name__)

# Load the saved ML model and preprocessor
saved_pipeline = joblib.load("netflix_model.pkl")

model = saved_pipeline["model"]
preprocessor = saved_pipeline["preprocessor"]

# Features must match the order used during training
features = [
    "release_year",
    "added_year",
    "added_month",
    "country_count",
    "genre_count",
    "description_length",
    "rating",
    "country",
    "listed_in"
]


# Display the homepage
@app.route("/")
def home():
    return render_template("index.html")


# Receive form data and make a prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data submitted by the HTML form
        release_year = int(request.form["release_year"])
        date_added = request.form["date_added"]
        rating = request.form["rating"]
        country = request.form["country"].strip()
        listed_in = request.form["listed_in"].strip()
        description = request.form["description"].strip()

        # Convert date into year and month
        added_date = datetime.strptime(date_added, "%Y-%m-%d")
        added_year = added_date.year
        added_month = added_date.month

        # Count countries and genres
        country_count = len([
            item for item in country.split(",") if item.strip()
        ])

        genre_count = len([
            item for item in listed_in.split(",") if item.strip()
        ])

        # Calculate description length
        description_length = len(description)

        # Prepare input in the same format as training data
        input_data = pd.DataFrame([{
            "release_year": release_year,
            "added_year": added_year,
            "added_month": added_month,
            "country_count": country_count,
            "genre_count": genre_count,
            "description_length": description_length,
            "rating": rating,
            "country": country,
            "listed_in": listed_in
        }], columns=features)

        # Apply the trained preprocessor
        input_processed = preprocessor.transform(input_data)

        # Predict the title type
        prediction = model.predict(input_processed)[0]

        # Show prediction on the webpage
        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as error:
        # Print the actual error in VS Code terminal
        print("Prediction Error:", repr(error))

        # Show the error on the webpage
        return render_template(
            "index.html",
            error=str(error)
        )


# Run the Flask application locally
if __name__ == "__main__":
    app.run(debug=True)