import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report

from data_loader import get_autoencoder_dataloaders
from model_autoencoder import MicrogridAutoencoder

def calculate_reconstruction_error(model, dataloader_or_tensor, criterion):
    """Passes data through the model and returns the MSE error for every single hour."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        if isinstance(dataloader_or_tensor, torch.Tensor):
            reconstructed = model(dataloader_or_tensor)
            for i in range(len(dataloader_or_tensor)):
                loss = criterion(reconstructed[i], dataloader_or_tensor[i])
                errors.append(loss.item())
        else:
            for batch in dataloader_or_tensor:
                reconstructed = model(batch)
                for i in range(len(batch)):
                    loss = criterion(reconstructed[i], batch[i])
                    errors.append(loss.item())
                    
    return np.array(errors)

def evaluate_model():
    print("\n--- Evaluating Unsupervised Autoencoder (The 59% Model) ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(current_dir, '..', 'data', 'rye_generation_and_load.csv')
    H5 = os.path.join(current_dir, '..', 'data', 'met_data.h5')
    
    # 1. Load Validation and Anomaly Data
    _, val_loader, anomaly_tensor, _, input_dim = get_autoencoder_dataloaders(CSV, H5)
    
    # 2. Load the trained model weights
    model = MicrogridAutoencoder(input_dim=input_dim)
    model_path = os.path.join(current_dir, '..', 'models', 'autoencoder_weights.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    criterion = nn.MSELoss()
    
    print("Calculating reconstruction errors for thresholding...")
    
    # 3. Calculate errors
    healthy_errors = calculate_reconstruction_error(model, val_loader, criterion)
    anomaly_errors = calculate_reconstruction_error(model, anomaly_tensor, criterion)
    
    # 4. Dynamic Thresholding (95th Percentile)
    threshold = np.percentile(healthy_errors, 95)
    print(f"\nDynamic Alarm Threshold calculated at MSE: {threshold:.4f}")
    
    # 5. Evaluation
    y_true = np.concatenate([np.zeros(len(healthy_errors)), np.ones(len(anomaly_errors))])
    y_pred_healthy = (healthy_errors > threshold).astype(int)
    y_pred_anomaly = (anomaly_errors > threshold).astype(int)
    y_pred = np.concatenate([y_pred_healthy, y_pred_anomaly])
    
    print("\nDeep Learning Classification Report (Autoencoder):")
    print(classification_report(y_true, y_pred, target_names=['Healthy Grid', 'Stressed/Anomaly Grid']))
    
    # 6. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(healthy_errors, bins=50, alpha=0.6, color='blue', label='Healthy Operations')
    plt.hist(anomaly_errors, bins=50, alpha=0.6, color='red', label='Grid Stress/Anomalies')
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Alarm Threshold ({threshold:.2f})')
    
    plt.title('Autoencoder Error Distribution: Healthy vs. Stressed Grid')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Number of Hours')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(current_dir, '..', 'models', 'error_distribution.png')
    plt.savefig(plot_path)
    print(f"\nVisualization saved to {plot_path}")

if __name__ == "__main__":
    evaluate_model()