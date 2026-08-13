import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import from our modular scripts
from data_loader import get_autoencoder_dataloaders
from model_autoencoder import MicrogridAutoencoder

def train_unsupervised_model():
    print("\n--- Training Unsupervised Autoencoder (The 59% Model) ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(current_dir, '..', 'data', 'rye_generation_and_load.csv')
    H5 = os.path.join(current_dir, '..', 'data', 'met_data.h5')
    
    # 1. Load Data
    train_loader, val_loader, anomaly_tensor, scaler, input_dim = get_autoencoder_dataloaders(CSV, H5)
    
    # 2. Initialize Model, Loss (MSE), and Optimizer
    model = MicrogridAutoencoder(input_dim=input_dim)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    print(f"Commencing Training for {epochs} Epochs...")
    
    # 3. The Epoch Loop
    for epoch in range(epochs):
        model.train() 
        running_train_loss = 0.0
        
        # Training Phase
        for batch in train_loader:
            optimizer.zero_grad()      
            reconstructed = model(batch) 
            loss = criterion(reconstructed, batch) 
            loss.backward()            
            optimizer.step()           
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        # Validation Phase
        model.eval() 
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    print("Training Complete!")
    
    # 4. Save the trained weights
    os.makedirs(os.path.join(current_dir, '..', 'models'), exist_ok=True)
    model_path = os.path.join(current_dir, '..', 'models', 'autoencoder_weights.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

if __name__ == "__main__":
    train_unsupervised_model()