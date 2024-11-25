# import numpy as np
# from flask import Flask, request, jsonify
# import pickle
# import logging
# from waitress import serve
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)

# # Load the trained Random Forest model
# model = pickle.load(open('/Users/vinay/Downloads/crop_price_app/linear_regression_model.pkl', 'rb'))

# # Load the OneHotEncoder and StandardScaler
# onehot_encoder = pickle.load(open('/Users/vinay/Downloads/crop_price_app/onehot_encoder.pkl', 'rb'))
# scaler = pickle.load(open('/Users/vinay/Downloads/crop_price_app/scaler.pkl', 'rb'))

# @app.route('/')
# def home():
#     app.logger.info("Home route accessed.")
#     return "Welcome to the Crop Price Prediction API!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract input data from the request
#     data = request.get_json()
#     app.logger.info(f"Received input data: {data}")

#     # Ensure all required fields are present
#     required_fields = [
#         'State', 'District', 'Market', 'Commodity', 
#         'Variety', 'Grade', 'Min Price', 'Max Price', 
#         'Day', 'Month', 'Year'
#     ]
#     for field in required_fields:
#         if field not in data:
#             return jsonify({'error': f'Missing required field: {field}'}), 400

#     # Extract values from the request data
#     state = data['State']
#     district = data['District']
#     market = data['Market']
#     commodity = data['Commodity']
#     variety = data['Variety']
#     grade = data['Grade']
#     min_price = data['Min Price']
#     max_price = data['Max Price']
#     day = data['Day']
#     month = data['Month']
#     year = data['Year']

#     # Create a DataFrame-like structure for OneHotEncoder
#     categorical_features = [[state, district, market, commodity, variety, grade]]
#     numerical_features = [[min_price, max_price, day, month, year]]

#     # Apply OneHotEncoder to categorical features
#     try:
#         categorical_encoded = onehot_encoder.transform(categorical_features).toarray()
#     except Exception as e:
#         app.logger.error(f"Error during one-hot encoding: {str(e)}")
#         return jsonify({'error': f'Error during one-hot encoding: {str(e)}'}), 400

#     # Scale the numerical features
#     try:
#         numerical_scaled = scaler.transform(numerical_features)
#     except Exception as e:
#         app.logger.error(f"Error during scaling: {str(e)}")
#         return jsonify({'error': f'Error during scaling: {str(e)}'}), 400

#     # Combine the encoded categorical and scaled numerical features
#     feature_vector_final = np.hstack([categorical_encoded, numerical_scaled])

#     # Make prediction using the trained model
#     try:
#         prediction = model.predict(feature_vector_final)
#     except Exception as e:
#         app.logger.error(f"Error during prediction: {str(e)}")
#         return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

#     # Return the prediction as a JSON response
#     return jsonify({'Predicted Modal Price': prediction[0]})

# if __name__ == "__main__":
#     app.logger.info("Starting the Flask app...")
#     serve(app, host='0.0.0.0', port=5001)


from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import logging
from waitress import serve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load the model and preprocessors
try:
    model = pickle.load(open('linear_regression_model.pkl', 'rb'))
    onehot_encoder = pickle.load(open('onehot_encoder.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the request
        data = request.get_json()
        
        # Ensure all required fields are present
        required_fields = [
            'State', 'District', 'Market', 'Commodity', 
            'Variety', 'Grade', 'Min Price', 'Max Price', 
            'Day', 'Month', 'Year'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Calculate additional features
        price_range = data['Max Price'] - data['Min Price']
        average_price = (data['Max Price'] + data['Min Price']) / 2

        # Prepare categorical and numerical features
        categorical_features = [[
            data['State'], data['District'], data['Market'],
            data['Commodity'], data['Variety'], data['Grade']
        ]]
        
        numerical_features = [[
            data['Min Price'], data['Max Price'],
            data['Day'], data['Month'], data['Year'],
            price_range, average_price
        ]]

        # Transform features
        categorical_encoded = onehot_encoder.transform(categorical_features).toarray()
        numerical_scaled = scaler.transform(numerical_features)

        # Combine features
        feature_vector = np.hstack([categorical_encoded, numerical_scaled])

        # Make prediction
        prediction = model.predict(feature_vector)

        return jsonify({'Predicted Modal Price': float(prediction[0])})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("Starting the Flask app...")
    serve(app, host='0.0.0.0', port=8080)
