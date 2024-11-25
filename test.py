import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model, encoders, and scaler
model = pickle.load(open('linear_regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
ohe = pickle.load(open('onehot_encoder.pkl', 'rb'))

# Example input data
data = {
    "State": "Gujarat",
    "District": "Amreli",
    "Market": "Damnagar",
    "Commodity": "Bhindi(Ladies Finger)",
    "Variety": "Bhindi",
    "Grade": "FAQ",
    "Min Price": 4100,
    "Max Price": 4500,
    "Day": 27,
    "Month": 7,
    "Year": 2023
}

# Add engineered features
price_range = data['Max Price'] - data['Min Price']
average_price = (data['Max Price'] + data['Min Price']) / 2

# Create a DataFrame-like array with all features
features = [
    data['State'], data['District'], data['Market'], data['Commodity'],
    data['Variety'], data['Grade'], data['Min Price'], data['Max Price'],
    data['Day'], data['Month'], data['Year'], price_range, average_price
]

# Split into categorical and numerical features
categorical_features = np.array(features[:6]).reshape(1, -1)  # Reshape to 2D
numerical_features = np.array(features[6:]).reshape(1, -1)  # Already 2D

# Apply one-hot encoding and scaling
categorical_encoded = ohe.transform(categorical_features).toarray()  # Ensure 2D
numerical_scaled = scaler.transform(numerical_features)  # Already 2D

# Combine features
feature_vector_final = np.hstack([categorical_encoded, numerical_scaled])  # Both are 2D now

# Make prediction
prediction = model.predict(feature_vector_final)

print(f"Predicted Modal Price: {prediction[0]}")
