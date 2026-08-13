import pandas as pd
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import os

def phase_0_extraction_and_merging(csv_path, h5_path):
    print("--- PHASE 0: Data Extraction, Stitching & Merging ---")
    
    # 1. Load Grid CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}. Please check the file path.")
    
    df_grid = pd.read_csv(csv_path)
    # Rename assuming the first column is the time index if 'index' isn't there
    if 'index' in df_grid.columns:
        df_grid = df_grid.rename(columns={'index': 'DateTime'})
    else:
        df_grid = df_grid.rename(columns={df_grid.columns[0]: 'DateTime'})
        
    df_grid['DateTime'] = pd.to_datetime(df_grid['DateTime'], utc=True).dt.tz_localize(None)
    
    # 2. Load HDF5 Weather Data
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}. Please check the file path.")
        
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
    
    # 3. Merge Datasets
    df_master = pd.merge(df_grid, df_weather, on='DateTime', how='inner')
    print(f"Merge complete! Final dataset shape: {df_master.shape}")
    return df_master


def generate_correlation_heatmap(df_master):
    print("--- PHASE 1: Generating Heatmap ---")
    
    # Drop DateTime to only keep numerical columns for correlation
    df_math = df_master.select_dtypes(include=[np.number])
    
    # To avoid a massive, unreadable grid, we dynamically select the first 2 grid columns 
    # and up to 5 weather columns.
    target_columns = df_math.columns[:7] 
    df_plot = df_math[target_columns]
    
    # Calculate Pearson correlation matrix
    corr_matrix = df_plot.corr()
    
    # Plotting parameters
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True,              
        cmap='coolwarm',         
        fmt=".2f",               
        linewidths=0.5,          
        cbar_kws={"shrink": .8}, 
        annot_kws={"size": 10}   
    )
    
    plt.title('Correlation Map: Grid Thermodynamics vs. Load', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save image
    file_name = 'dataset_visualization.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"SUCCESS: Image saved locally as '{file_name}'")
    plt.close()


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Replace these strings with the EXACT path on your computer.
    # Example for Windows: r'C:\Users\YourName\Documents\Project\rye_generation_and_load.csv'
    # Example for Mac/Linux/Colab: '/content/rye_generation_and_load.csv'
    
    CSV_FILENAME = r'"E:\Python_Course\Chapter 1\iems_project\data\rye_generation_and_load.csv"'
    H5_FILENAME = r'"E:\Python_Course\Chapter 1\iems_project\data\met_data.h5"'
    
    try:
        # Run extraction
        master_dataframe = phase_0_extraction_and_merging(CSV_FILENAME, H5_FILENAME)
        # Run image generation
        generate_correlation_heatmap(master_dataframe)
    except Exception as e:
        print(f"AN ERROR OCCURRED: {e}")