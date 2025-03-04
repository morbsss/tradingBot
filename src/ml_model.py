import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import yaml
from src.utils import create_sequences

def load_model_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def train_ml_model(features, target, sequence_length):
    config = load_model_config()
    
    # Convert to numpy and ensure target is 1D
    features_np = features.values
    target = target.ravel() if target.ndim > 1 else target
    
    # Create sequences
    X = create_sequences(features_np, sequence_length)
    y = target[sequence_length:]
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(
        units=config["LSTM"]["lstm_units"],
        return_sequences=False,  # Only return final output
        input_shape=(sequence_length, X.shape[2])  # (timesteps, features)
    ))
    model.add(Dropout(0.2))  # Prevent overfitting
    model.add(Dense(units=config["LSTM"]["dense_units"], activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))  # Binary classification
    
    # Compile model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=config["LSTM"]["epochs"],
        batch_size=config["LSTM"]["batch_size"],
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Save model
    model.save("models/trained_models/lstm_model.keras")
    return model

def predict(model, features, sequence_length):
    """
    Predict using the trained LSTM model.
    
    Args:
        model: Trained Keras LSTM model.
        features (pd.DataFrame): Features to predict on.
        sequence_length (int): Number of time steps per sequence.
    
    Returns:
        np.ndarray: Predicted probabilities or binary labels.
    """
    # Convert to sequences
    X = create_sequences(features.values, sequence_length)

    # Predict
    predictions = model.predict(X, verbose=0)
    return (predictions > 0.5).astype(int).ravel()  # Binary output