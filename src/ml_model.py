import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import yaml
from src.utils import create_sequences

def load_model_config():
    with open("src/config.yaml", "r") as file:
        return yaml.safe_load(file)

def train_ml_model(features, target, ml_model, sequence_length=None):
    """
    Train either an LSTM or XGBoost model.
    
    Args:
        features (pd.DataFrame): Features (e.g., MACD, RSI, sentiment, Fibonacci).
        target (np.ndarray): Target (e.g., price increase binary).
        model_type (str): "LSTM" or "XGBoost".
        sequence_length (int): Number of time steps for LSTM (ignored for XGBoost).
    
    Returns:
        model: Trained model (Keras model or XGBoost model).
    """
    config = load_model_config()

    if ml_model == 'LSTM':    
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
    
    elif ml_model == "XGBoost":
        # Use features directly (no sequences needed for XGBoost)
        X = features.values
        y = target.ravel() if target.ndim > 1 else target
        
        # Split into train and test sets
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train XGBoost model
        model = XGBClassifier(
            max_depth=config["xgboost"]["max_depth"],
            learning_rate=config["xgboost"]["learning_rate"],
            n_estimators=config["xgboost"]["n_estimators"],
            subsample=config["xgboost"]["subsample"],
            colsample_bytree=config["xgboost"]["colsample_bytree"],
            random_state=config["xgboost"]["random_state"],
            eval_metric="logloss"     # Binary classification metric
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=1)
        joblib.dump(model, "models/trained_models/xgb_model.pkl")
        return model

def predict(model, features, ml_model, sequence_length=None):
    """
    Predict using the trained LSTM model.
    
    Args:
        model: Trained Keras LSTM model.
        features (pd.DataFrame): Features to predict on.
        sequence_length (int): Number of time steps per sequence.
    
    Returns:
        np.ndarray: Predicted probabilities or binary labels.
    """
    if ml_model == "LSTM":
        # Convert to sequences
        X = create_sequences(features.values, sequence_length)

        # Predict
        predictions = model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).ravel()  # Binary output

    elif ml_model == "XGBoost":
        X = features.values
        predictions = model.predict_proba(X)[:, 1]  # Probability of class 1 (price increase)
        return (predictions > 0.5).astype(int)