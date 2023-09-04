import requests
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

API_KEY = "9iQ9FQuwOqoRtaBJRa6psl7PgG4TYHLISVvXrBxbWJ-v"
DEPLOYMENT_URL = "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/84edf2bc-d756-4c4a-b82a-a60f0ee95a6d/predictions?version=2021-05-01"

# Load the scaler for preprocessing of data
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)


def score_model(input_data):
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={
        "apikey": API_KEY,
        "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'
    })
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    payload_scoring = {"input_data": input_data}

    response_scoring = requests.post(DEPLOYMENT_URL, json=payload_scoring,
                                     headers={'Authorization': 'Bearer ' + mltoken})
    return response_scoring.json()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    input_features = {
        "gender": int(request.form["gender"]),
        "tenthboard": int(request.form["tenthboard"]),
        "tenthmarks": float(request.form["tenthmarks"]),
        "twelvethboard": int(request.form["twelvethboard"]),
        "twelvethmarks": float(request.form["twelvethmarks"]),
        "stream": int(request.form["stream"]),
        "cgpa": float(request.form["cgpa"]),
        "internship": int(request.form["internship"]),
        "training": int(request.form["training"]),
        "Backlog": int(request.form["Backlog"]),
        "Project": int(request.form["Project"]),
        "communication": int(request.form["communication"]),
        "courses": int(request.form["courses"])
    }
    required_features = ['gender', 'tenthboard', 'tenthmarks', 'twelvethboard', 'twelvethmarks', 'stream', 'cgpa',
                         'internship', 'training', 'Backlog', 'Project', 'communication', 'courses']

    values_required = [input_features[feature] for feature in required_features]
    final_features = np.array(values_required).reshape(1, -1)
    scaled_input = loaded_scaler.transform(final_features)

    payload = {"fields": required_features, "values": [list(scaled_input[0])]}
    prediction_response = score_model([payload])  # Remove the extra list here

    print("Prediction response:", prediction_response)

    # Check the structure of the response and adjust the following code accordingly
    # prediction_perc = prediction_response['predictions'][0]['values'][0][0]['prediction'] * 100
    # prediction_perc = prediction_response['predictions'][0]['values'][0][1] * 100
    # Extract the prediction value
    prediction_value = prediction_response['predictions'][0]['values'][0][1][1]

    # Convert the prediction value to percentage
    prediction_perc = round(prediction_value * 100,3)

    return render_template('index.html', prediction=prediction_perc)


@app.route("/clear", methods=["GET", "POST"])
def clear():
    prediction = ""
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
