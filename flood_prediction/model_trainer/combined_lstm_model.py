# combined_lstm_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

def prepare_combined_data(data, look_back=7):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0]) 
    return np.array(X), np.array(y)

def train_combined_lstm_model(river_data, weather_data):
    if len(river_data) < 8 or len(weather_data) < 8:
        raise ValueError("Not enough data to train the model. Need at least 8 data points.")

    combined_data = pd.merge(river_data, weather_data, on='date', how='inner')
    combined_data = combined_data.sort_values('date')

    features = ['river_discharge', 'temperature', 'humidity', 'pressure']
    data = combined_data[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = prepare_combined_data(scaled_data)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences. Need more data points.")

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)

    return model, scaler

def save_combined_model_and_scaler(model, scaler, model_filename, scaler_filename):
    """Save both the model and the scaler."""
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)

def load_combined_model_and_scaler(model_filename, scaler_filename):
    """Load both the model and the scaler."""
    model = tf.keras.models.load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    return model, scaler

def make_combined_prediction(model, scaler, input_data):
    """Make predictions using the trained model and scaler."""
    scaled_input = scaler.transform(input_data)
    scaled_prediction = model.predict(scaled_input.reshape(1, -1, input_data.shape[1]))
    
    # Create a dummy array with the same shape as the original input
    dummy_array = np.zeros((1, 4))
    dummy_array[0, 0] = scaled_prediction[0, 0]  # Set the first column to the predicted value
    
    # Inverse transform the dummy array
    inverse_transformed = scaler.inverse_transform(dummy_array)
    
    # Return only the predicted river discharge value
    return inverse_transformed[0, 0]