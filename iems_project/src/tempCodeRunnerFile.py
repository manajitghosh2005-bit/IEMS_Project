import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics import classification_report

# Import from our modular scripts
from data_loader import get_supervised_dataloaders
from model_mlp import MicrogridClassifier

def train_and_evaluate():
    print("\n--- Training & Evaluating Supervised MLP (The 99% Model) ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(current_dir, '..', 'data', 'rye_generation_and_load.csv')
    H5 = os.path.join(current_dir, '..', 'data', 'met_data.h5')
    
    # 1. Load the data
    train_loader, test_loader, input_dim = get_supervised_dataloaders(CSV, H5)
    
    # 2. Initialize Model, Loss (CrossEntropy), and Optimizer
    model = MicrogridClassifier(input_dim=input_dim, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    print(f"\nTraining Supervised Deep Classifier for {epochs} Epochs...")
    
    # 3. The Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels) # CrossEntropy classification error
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {running_loss/len(train_loader):.4f}")
            
    print("\nTraining Complete! Deploying model on unseen test data...")
    
    # 4. The Evaluation Phase
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # The network outputs 3 probabilities. torch.max picks the highest one.
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())
            
    # 5. The Final Verdict
    print("\nDeep Learning Classification Report (Supervised MLP):")
    # Using digits=4 to see exactly how much we beat the LDA by
    print(classification_report(all_targets, all_preds, digits=4, target_names=['State 0', 'State 1', 'State 2']))
    
    # 6. Save the trained weights
    os.makedirs(os.path.join(current_dir, '..', 'models'), exist_ok=True)
    model_path = os.path.join(current_dir, '..', 'models', 'mlp_weights.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel weights saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()