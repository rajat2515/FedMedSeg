import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm # For nice progress bars

# --- 1. CONFIGURATION ---

# Set paths relative to 'notebooks/' directory
PROCESSED_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
RESULTS_DIR = Path("../results")
SRC_DIR = Path("../src")

# Add 'src' to system path to import our Dataset
sys.path.append(str(SRC_DIR))
try:
    from data_preprocessing import PneumoniaDataset
    print("Successfully imported PneumoniaDataset.")
except ImportError:
    print("ERROR: Could not import PneumoniaDataset from src/data_preprocessing.py")

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 25 # Set a max; EarlyStopping will find the best
EARLY_STOP_PATIENCE = 5
MODEL_SAVE_PATH = MODELS_DIR / "model_1_basic_cnn_best.pth"
HISTORY_SAVE_PATH = RESULTS_DIR / "model_1_history.json"

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Model1_BasicCNN(nn.Module):
    def __init__(self):
        super(Model1_BasicCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 224 -> 112
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 112 -> 56
        
        # Classifier Head
        self.flatten = nn.Flatten()
        # Calculate flattened size: 64 channels * 56x56 pixels
        self.fc1 = nn.Linear(in_features=64 * 56 * 56, out_features=64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=64, out_features=1) # Output 1 logit for binary classification

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the device
model = Model1_BasicCNN().to(device)
print(model)

# Test with a dummy input
try:
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    output = model(dummy_input)
    print(f"\nSuccess! Output shape: {output.shape}") # Should be [1, 1]
except Exception as e:
    print(f"\nError during model test: {e}")
    # Define transforms
# Training transforms include augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), # Augmentation
    transforms.RandomRotation(10),     # Augmentation
    transforms.ToTensor(),             # Normalizes to [0, 1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Optional: ImageNet stats
])

# Validation transforms do NOT include augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create Datasets
train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
val_df = pd.read_csv(PROCESSED_DIR / "val_split.csv")

train_dataset = PneumoniaDataset(PROCESSED_DIR / "train_split.csv", transform=train_transform)
val_dataset = PneumoniaDataset(PROCESSED_DIR / "val_split.csv", transform=val_transform)

# Create DataLoaders
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"DataLoaders created.")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# --- Calculate Weighted Loss ---
all_data_df = pd.concat([train_df, val_df])
normal_count = (all_data_df['label'] == 0).sum()
pneumonia_count = (all_data_df['label'] == 1).sum()

# pos_weight = count of negative samples / count of positive samples
pos_weight = torch.tensor(normal_count / pneumonia_count, dtype=torch.float32).to(device)
print(f"Imbalance ratio (Pneumonia:Normal) = {pneumonia_count/normal_count:.2f}:1")
print(f"Calculated 'pos_weight' for loss function: {pos_weight.item():.4f}")

# Define Loss and Optimizer ("Compile" step)
# We use BCEWithLogitsLoss because it's numerically stable 
# and our model outputs raw logits (not a sigmoid).
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_val_loss = float('inf')
epochs_no_improve = 0

history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

print("Starting training...")

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train() # Set model to training mode
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # Use tqdm for a progress bar
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for inputs, labels in train_pbar:
        # Move data to the device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss (squeeze outputs and make labels float)
        # We use labels.float() because BCEWithLogitsLoss expects float targets
        squeezed_outputs = outputs.squeeze()
        float_labels = labels.float() 
        loss = criterion(squeezed_outputs, float_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # --- ADD THIS FOR ACCURACY ---
        preds = torch.sigmoid(squeezed_outputs) > 0.5
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        # -----------------------------
        
        train_pbar.set_postfix({"loss": loss.item()})
        
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / train_total
    
    # --- Validation Phase ---
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad(): # Disable gradient calculation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            # Squeeze outputs and make labels float
            squeezed_outputs = outputs.squeeze()
            float_labels = labels.float()
            loss = criterion(squeezed_outputs, float_labels)
            val_loss += loss.item()

            # --- ADD THIS FOR ACCURACY ---
            preds = torch.sigmoid(squeezed_outputs) > 0.5
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            # -----------------------------
            
            val_pbar.set_postfix({"loss": loss.item()})
            
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_correct / val_total
    
    # Print all metrics
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
     # --- ADD THIS TO SAVE HISTORY ---
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)
     # --------------------------------

    # --- ModelCheckpoint Logic ---
    if avg_val_loss < best_val_loss:
        print(f"Validation loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_val_loss = avg_val_loss
        epochs_no_improve = 0 # Reset counter
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")
        
    # --- EarlyStopping Logic ---
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print(f"\nTraining finished. Best model saved to {MODEL_SAVE_PATH}")
# --- ADD THIS TO SAVE THE JSON FILE ---
import json
print(f"Saving training history to {HISTORY_SAVE_PATH}")
with open(HISTORY_SAVE_PATH, 'w') as f:
    json.dump(history, f, indent=4)
print("History saved successfully.")
# -------------------------------------