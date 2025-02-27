import numpy as np

def create_sequences(data, sequence_length, target_col_idx=None):
    """
    Convert a DataFrame into sequences for RNN input.
    
    Args:
        data (np.ndarray): Input data (e.g., features and optionally target).
        sequence_length (int): Number of time steps per sequence.
        target_col_idx (int, optional): Index of target column if included in data.
    
    Returns:
        X (np.ndarray): 3D array of shape (samples, sequence_length, features).
        y (np.ndarray, optional): 1D array of targets if target_col_idx is provided.
    """
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Sequence of features
        if target_col_idx is not None:
            y.append(data[i + sequence_length, target_col_idx])  # Next target value
    X = np.array(X)
    if target_col_idx is not None:
        return X, np.array(y)
    print(X)
    return X