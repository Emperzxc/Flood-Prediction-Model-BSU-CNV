from flask import Flask, jsonify , request
from flask_restful import Resource, Api
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

# Load the trained model
model = pickle.load(open('Flood-Prediction-Model-CNV-050124 (1).pkl', 'rb'))


# Define a route to handle predictions on CSV files
@app.route('/data', methods = ['GET'])
def predict_from_csv():
    # Read the CSV file from the working directory
    file_path = 'test.csv'

    # Check if the file is CSV
    if file_path.endswith('.csv'):
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Make predictions
        predictions = model.predict(data)

        # Prepare the predictions as JSON
        results = {'predictions': predictions.tolist()}  # Convert to list for JSON compatibility

        return jsonify(results)
    else:
        return jsonify({'error': 'Invalid file format. Please make sure the input file is in CSV format.'})


@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json

    if request_data is None:
        return jsonify({'error': 'No data provided'})

    predictions = []
    for input_data in request_data:
        df = pd.DataFrame([input_data],
                          columns=['MONTH', 'RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION',
                                   'P_RAINFALL', 'P_TMAX', 'P_TMIN', 'P_RH', 'P_WIND_SPEED',
                                   'P_WIND_DIRECTION', 'Name of Barangay_encoded'])

        prediction = model.predict(df)
        predictions.append(int(prediction[0]))

    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run(debug=True)
