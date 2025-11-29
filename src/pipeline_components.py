import kfp.dsl as dsl
import kfp.components as components
import os
import json
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Use Regressor for Boston regression task
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import dvc.api

@dsl.component
def data_extraction(output_path: dsl.OutputPath(str)) -> str:
    """Fetch versioned dataset using DVC."""
    # Assuming the repo is the current directory and DVC is set up.
    # For remote fetch, use dvc.api.get(url='https://github.com/yourusername/mlops-kubeflow-assignment.git', path='data/raw_data.csv', rev='main', out=output_path)
    # But for simplicity, assume dvc pull if needed.
    subprocess.run(['dvc', 'pull', 'data/raw_data.csv'], check=True)
    raw_data_path = 'data/raw_data.csv'
    return raw_data_path

@dsl.component
def data_preprocessing(input_path: str, train_x_path: dsl.OutputPath(str), train_y_path: dsl.OutputPath(str), test_x_path: dsl.OutputPath(str), test_y_path: dsl.OutputPath(str)):
    """Preprocess the data: clean, scale, split."""
    df = pd.read_csv(input_path)
    # Assuming columns: CRIM, ZN, ..., MEDV as target (rename if needed)
    if 'MEDV' in df.columns:
        df = df.rename(columns={'MEDV': 'target'})
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    np.save(train_x_path, X_train_scaled)
    np.save(train_y_path, y_train)
    np.save(test_x_path, X_test_scaled)
    np.save(test_y_path, y_test)

@dsl.component
def model_training(train_x_path: str, train_y_path: str, model_path: dsl.OutputPath(str)):
    """Train a Random Forest Regressor model."""
    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

@dsl.component
def model_evaluation(test_x_path: str, test_y_path: str, model_path: str, metrics_path: dsl.OutputPath(str)):
    """Evaluate the model and save metrics."""
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {'mse': mse, 'r2': r2}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

# Compile components to YAML (run this locally to generate components/*.yaml)
if __name__ == '__main__':
    components.create_component_from_func(
        data_extraction,
        output_component_file='components/data_extraction.yaml',
        packages_to_install=['dvc', 'subprocess']
    )
    components.create_component_from_func(
        data_preprocessing,
        output_component_file='components/data_preprocessing.yaml',
        packages_to_install=['pandas', 'numpy', 'scikit-learn']
    )
    components.create_component_from_func(
        model_training,
        output_component_file='components/model_training.yaml',
        packages_to_install=['numpy', 'scikit-learn', 'joblib']
    )
    components.create_component_from_func(
        model_evaluation,
        output_component_file='components/model_evaluation.yaml',
        packages_to_install=['numpy', 'scikit-learn', 'joblib', 'json']
    )