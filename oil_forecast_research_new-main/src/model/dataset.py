import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class PetroleumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class DataProcessor:
    def __init__(self, seq_len=30, horizon=5):
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, df: pd.DataFrame, target_cols: list, feature_cols: list, is_train=True):
        features_data = df[feature_cols].values
        targets_data = df[target_cols].values
        
        if is_train:
            features_scaled = self.feature_scaler.fit_transform(features_data)
            targets_scaled = self.target_scaler.fit_transform(targets_data)
        else:
            features_scaled = self.feature_scaler.transform(features_data)
            targets_scaled = self.target_scaler.transform(targets_data)
            
        X, y = [], []
        valid_range = len(df) - self.seq_len - self.horizon + 1
        for i in range(valid_range):
            X.append(features_scaled[i : i + self.seq_len])
            y.append(targets_scaled[i + self.seq_len : i + self.seq_len + self.horizon])
        return np.array(X), np.array(y)
    
    def inverse_transform_targets(self, y_scaled):
        shape = y_scaled.shape
        if len(shape) == 3:
            y_flat = y_scaled.reshape(-1, shape[-1])
            y_inv = self.target_scaler.inverse_transform(y_flat)
            return y_inv.reshape(shape)
        return self.target_scaler.inverse_transform(y_scaled)