import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

def prepare_data(data, look_back=7):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def train_lstm_model(data):
    if len(data) < 8:
        raise ValueError("Not enough data to train the model. Need at least 8 data points.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = prepare_data(scaled_data)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences. Need more data points.")

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)

    return model, scaler

def save_model_and_scaler(model, scaler, model_filename, scaler_filename):
    """Save both the model and the scaler."""
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)

def load_model_and_scaler(model_filename, scaler_filename):
    """Load both the model and the scaler."""
    model = tf.keras.models.load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    return model, scaler

def make_prediction(model, scaler, input_data):
    """Make predictions using the trained model and scaler."""
    scaled_input = scaler.transform(input_data.reshape(-1, 1))
    scaled_prediction = model.predict(scaled_input.reshape(1, -1, 1))
    return scaler.inverse_transform(scaled_prediction)[0, 0]








