import torch
import torch.nn as nn

class MicrogridAutoencoder(nn.Module):
    def __init__(self, input_dim=13):
        super(MicrogridAutoencoder, self).__init__()
        
        # --- ENCODER: Compressing the grid physics ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.1), # 10% dropout to prevent overfitting
            
            # The Latent Space Bottleneck (3 dimensions)
            nn.Linear(8, 3),
            nn.ReLU()
        )
        
        # --- DECODER: Reconstructing the grid physics ---
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Expanding back to the original 13 sensors
            nn.Linear(8, input_dim) 
        )

    def forward(self, x):
        # 1. Pass input through the encoder to get compressed state
        latent_space = self.encoder(x)
        
        # 2. Pass compressed state through decoder to rebuild input
        reconstructed = self.decoder(latent_space)
        
        return reconstructed

if __name__ == "__main__":
    model = MicrogridAutoencoder(input_dim=13)
    print("Autoencoder Architecture successfully initialized:")
    print(model)