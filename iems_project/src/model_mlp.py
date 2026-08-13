import torch
import torch.nn as nn

class MicrogridClassifier(nn.Module):
    def __init__(self, input_dim=13, num_classes=3):
        super(MicrogridClassifier, self).__init__()
        
        # A deep, non-linear Feedforward Neural Network (Multi-Layer Perceptron)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),   # Expand the 13 features to 32 nodes
            nn.ReLU(),
            nn.Dropout(0.2),            # 20% Dropout to aggressively prevent memorization
            
            nn.Linear(32, 16),          # Compress to 16 nodes
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, num_classes)  # Final output: 3 nodes (one for each Grid State)
        )

    def forward(self, x):
        # The output will be 3 raw numbers (logits). 
        # The highest number represents the network's final prediction.
        return self.network(x)

if __name__ == "__main__":
    model = MicrogridClassifier()
    print("Supervised MLP Classifier successfully initialized:")
    print(model)