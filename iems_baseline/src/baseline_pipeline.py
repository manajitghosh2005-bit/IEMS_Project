import pandas as pd
import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def phase_0_extraction_and_merging(csv_path, h5_path):
    print("--- PHASE 0: Data Extraction, Stitching & Merging ---")
    df_grid = pd.read_csv(csv_path)
    df_grid = df_grid.rename(columns={'index': 'DateTime'})
    df_grid['DateTime'] = pd.to_datetime(df_grid['DateTime'], utc=True).dt.tz_localize(None)
    
    sensor_series_dict = {}
    with h5py.File(h5_path, 'r') as hdf:
        base_group = hdf['lat63_41_lon10_11']
        for sensor in base_group.keys():
            forecast_group = base_group[sensor]['forecast']
            all_times = []
            all_values = []
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
    df_master = pd.merge(df_grid, df_weather, on='DateTime', how='inner')
    return df_master

def phase_2_clean_and_scale(df_master):
    print("\n--- PHASE 1 & 2: Data Cleaning & Feature Scaling ---")
    df_clean = df_master.dropna().copy()
    
    df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['DateTime'].dt.hour / 24)
    df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['DateTime'].dt.hour / 24)
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['DateTime'].dt.month / 12)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['DateTime'].dt.month / 12)
    
    df_math = df_clean.drop(columns=['DateTime'])
    
    core_features = df_math.columns
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_math)
    
    df_scaled = pd.DataFrame(scaled_array, columns=core_features, index=df_clean.index)
    return df_clean, df_scaled

def phase_3_pca(df_scaled):
    print("\n--- PHASE 3: Principal Component Analysis (PCA) ---")
    pca = PCA(n_components=3)
    pca_array = pca.fit_transform(df_scaled)
    
    pca_columns = ["Master_Factor_1", "Master_Factor_2", "Master_Factor_3"]
    df_pca = pd.DataFrame(pca_array, columns=pca_columns, index=df_scaled.index)
    return df_pca

def phase_4_kmeans(df_pca, df_clean):
    print("\n--- PHASE 4: Unsupervised State Discovery (K-Means) ---")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_pca)
    
    df_final = df_clean.copy()
    df_final['Grid_State'] = cluster_labels
    return df_final

def phase_5_profile_and_predict(df_final, df_pca):
    print("\n--- PHASE 5: Centroid Profiling & LDA Classification ---")
    
    # 1. Profiling the Physical Centroids
    print("Extracting physical averages for each Grid State...\n")
    core_cols = ['Consumption', 'Solar', 'Wind', 'air_temperature_2m']
    profile = df_final.groupby('Grid_State')[core_cols].mean()
    print(profile)
    
    # 2. Supervised Learning: Linear Discriminant Analysis (LDA)
    print("\nTraining Supervised LDA Classifier...")
    
    # Inputs (X) = PCA Factors, Output (y) = The Grid State
    X = df_pca
    y = df_final['Grid_State']
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    # Test the LDA on unseen data
    y_pred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nLDA Model Accuracy on Unseen Data: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return lda

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(current_dir, '..', 'data', 'rye_generation_and_load.csv')
    H5_PATH = os.path.join(current_dir, '..', 'data', 'met_data.h5')
    
    # 1. Pipeline Execution
    master_data = phase_0_extraction_and_merging(CSV_PATH, H5_PATH)
    clean_data, scaled_data = phase_2_clean_and_scale(master_data)
    pca_data = phase_3_pca(scaled_data)
    labeled_data = phase_4_kmeans(pca_data, clean_data)
    
    # 2. Analysis & Prediction
    lda_model = phase_5_profile_and_predict(labeled_data, pca_data)