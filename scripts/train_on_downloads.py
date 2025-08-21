# Auto-training script for downloaded Binance historic data
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Configuration
DATA_DIR = 'artifacts/binance_downloads'
MODEL_DIR = 'artifacts/models'
SUMMARY_DIR = 'artifacts/training'

# LSTM Model Definition
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_all_csv(data_dir):
    """Loads all CSVs from a directory and returns a dictionary of path -> dataframe."""
    datasets = {}
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return datasets
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv'):
                path = os.path.join(root, f)
                try:
                    df = pd.read_csv(path)
                    datasets[path] = df
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    return datasets

def train_lstm_on_df(df, pair_name):
    """Trains an LSTM model on the dataframe, returns path and metrics."""
    sequence_length = 50
    data = df['close'].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

    if not X:
        return None, None

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LSTMNet(input_size=1, hidden_size=32, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5): # small example epoch count
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # Validation
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_filename = f'lstm_model_{pair_name}_{timestamp}.pt'
    model_path = os.path.join(MODEL_DIR, model_filename)
    torch.save(model.state_dict(), model_path)

    metrics = {
        'validation_loss': val_loss,
        'num_samples': len(X),
        'num_features': 1,
        'features_used': ['close_sequence']
    }
    return model_path, metrics

def train_xgboost_on_df(df, pair_name):
    """Trains an XGBoost model on the dataframe, returns path and metrics."""
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    features = ['open', 'high', 'low', 'close', 'volume']
    X = df[features].values
    y = df['target'].values

    if len(X) == 0:
        return None, None

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_filename = f'xgb_model_{pair_name}_{timestamp}.json'
    model_path = os.path.join(MODEL_DIR, model_filename)
    model.save_model(model_path)

    metrics = {
        'validation_rmse': rmse,
        'num_samples': len(X),
        'num_features': len(features),
        'features_used': features
    }
    return model_path, metrics

def write_training_summary(summary_data):
    """Writes a summary of the training process to a JSON file."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary_path = os.path.join(SUMMARY_DIR, 'training_summary.json')

    tmp_path = summary_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    os.replace(tmp_path, summary_path)

    print(f"Training summary written to {summary_path}")

def run_training():
    """Main function to run the training pipeline."""
    start_time = datetime.now()
    print(f"Starting training run at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    datasets = load_all_csv(DATA_DIR)
    if not datasets:
        print("No CSV files found to train on. Exiting.")
        return

    training_results = []

    for path, df in datasets.items():
        pair_name = os.path.basename(path).replace('.csv', '')
        print(f"\\n--- Processing {pair_name} ---")

        # Train LSTM
        print("Training LSTM model...")
        lstm_path, lstm_metrics = train_lstm_on_df(df.copy(), pair_name)
        if lstm_path:
            training_results.append({
                'pair': pair_name,
                'model_type': 'LSTM',
                'model_path': lstm_path,
                'metrics': lstm_metrics
            })
            print(f"LSTM model saved to {lstm_path}")

        # Train XGBoost
        print("Training XGBoost model...")
        xgb_path, xgb_metrics = train_xgboost_on_df(df.copy(), pair_name)
        if xgb_path:
            training_results.append({
                'pair': pair_name,
                'model_type': 'XGBoost',
                'model_path': xgb_path,
                'metrics': xgb_metrics
            })
            print(f"XGBoost model saved to {xgb_path}")

    end_time = datetime.now()

    summary = {
        'run_start_time': start_time.isoformat(),
        'run_end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'models_trained_count': len(training_results),
        'training_details': training_results
    }

    write_training_summary(summary)

    print(f"\\nTraining run finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_training()
