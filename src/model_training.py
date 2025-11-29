# This can be a standalone training script, but since components use functions, this might be optional or for local testing.
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(train_x_path, train_y_path, model_path):
    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

if __name__ == '__main__':
    # Example usage
    train_model('x_train.npy', 'y_train.npy', 'model.pkl')