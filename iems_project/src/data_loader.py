import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ==========================================
# 1. THE COMMON DATA EXTRACTION ENGINE
# ==========================================
def extract_and_engineer_data(csv_path, h5_path):
    print("Extracting and stitching raw SCADA/Meteorological data...")
    
    # Load CSV
    df_grid = pd.read_csv(csv_path)
    df_grid = df_grid.rename(columns={'index': 'DateTime'})
    df_grid['DateTime'] = pd.to_datetime(df_grid['DateTime'], utc=True).dt.tz_localize(None)
    
    # Load and Stitch HDF5
    sensor_series_dict = {}
    with h5py.File(h5_path, 'r') as hdf:
        base_group = hdf['lat63_41_lon10_11']
        for sensor in base_group.keys():
            forecast_group = base_group[sensor]['forecast']
            all_times, all_values = [], []
            for forecast_time in forecast_group.keys():
                data_node = forecast_group[forecast_time]
                all_times.append(data_node['axis1'][:])
                all_values.append(data_node['block0_values'][:, 0]) 
            
            combined_times = np.concatenate(all_times)
            combined_values = np.concatenate(all_values)
            temp_df = pd.DataFrame({
                'DateTime': pd.to_datetime(combined_times, unit='ns', utc=True).tz_localize(None),
                sensor: combined_values
            })
            temp_df = temp_df.drop_duplicates(subset='DateTime', keep='first')
            sensor_series_dict[sensor] = temp_df.set_index('DateTime')[sensor]
            
    df_weather = pd.DataFrame(sensor_series_dict).reset_index()
    df_master = pd.merge(df_grid, df_weather, on='DateTime', how='inner').dropna()
    
    # Cyclical Time Features
    df_master['hour_sin'] = np.sin(2 * np.pi * df_master['DateTime'].dt.hour / 24)
    df_master['hour_cos'] = np.cos(2 * np.pi * df_master['DateTime'].dt.hour / 24)
    df_master['month_sin'] = np.sin(2 * np.pi * df_master['DateTime'].dt.month / 12)
    df_master['month_cos'] = np.cos(2 * np.pi * df_master['DateTime'].dt.month / 12)
    
    return df_master

# ==========================================
# 2. AUTOENCODER DATA PREP (THE 59% MODEL)
# ==========================================
class MicrogridDataset(Dataset):
    """Custom PyTorch Dataset for Unlabeled Autoencoder training."""
    def __init__(self, features_array):
        self.x = torch.tensor(features_array, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx]

def get_autoencoder_dataloaders(csv_path, h5_path, batch_size=64):
    print("\n--- Prepping Data for Semi-Supervised Autoencoder ---")
    df_master = extract_and_engineer_data(csv_path, h5_path)
    
    # Define physical anomaly thresholds to separate "Healthy" from "Stress"
    is_anomaly = (df_master['air_temperature_2m'] < 278) | (df_master['Consumption'] > 20)
    
    df_healthy = df_master[~is_anomaly].drop(columns=['DateTime'])
    df_anomalies = df_master[is_anomaly].drop(columns=['DateTime'])
    
    # Scale based strictly on healthy data
    scaler = StandardScaler()
    train_data, val_data = train_test_split(df_healthy, test_size=0.2, random_state=42)
    
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    anomalies_scaled = scaler.transform(df_anomalies) 
    
    train_loader = DataLoader(MicrogridDataset(train_scaled), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MicrogridDataset(val_scaled), batch_size=batch_size, shuffle=False)
    anomaly_tensor = torch.tensor(anomalies_scaled, dtype=torch.float32)
    
    return train_loader, val_loader, anomaly_tensor, scaler, train_scaled.shape[1]

# ==========================================
# 3. MLP CLASSIFIER DATA PREP (THE 99% MODEL)
# ==========================================
def get_supervised_dataloaders(csv_path, h5_path, batch_size=64):
    print("\n--- Prepping Data for Supervised MLP Classifier ---")
    df_master = extract_and_engineer_data(csv_path, h5_path)
    df_math = df_master.drop(columns=['DateTime'])
    
    # Scale entire dataset
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_math)
    
    # Generate the "Answer Key" using PCA and K-Means
    pca = PCA(n_components=3)
    pca_array = pca.fit_transform(scaled_array)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_array)
    
    # Split 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_array, labels, test_size=0.2, random_state=42
    )
    
    # Wrap in PyTorch TensorDatasets & DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaled_array.shape[1]