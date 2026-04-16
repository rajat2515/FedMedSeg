import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import torchmetrics # For calculating metrics
import time
import os

# --- Import the model ---
# Imports from your model2.py file
from model2 import DeeperCNN

# --- Configuration (Global Constants) ---
TRAIN_DIR = 'data/raw/chest_xray/train' # Corrected path
VAL_DIR = 'data/raw/chest_xray/val'     # Corrected path
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = 'model_2_best.pth'

# --- 1. Define Data Transforms (Global) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet std
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# --- 4. Training Loop Definition ---
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    start_time = time.time()
    
    # Trackers for best model
    best_model_weights = model.state_dict()
    best_f1 = 0.0

    # Initialize torchmetrics for metrics
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # Reset metrics
            f1_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            accuracy_metric.reset()

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1) # Ensure correct shape/type

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Apply sigmoid to outputs for metrics
                    preds = torch.sigmoid(outputs)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                # Update metrics
                f1_metric.update(preds, labels)
                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                accuracy_metric.update(preds, labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_metric.compute()
            epoch_f1 = f1_metric.compute()
            epoch_precision = precision_metric.compute()
            epoch_recall = recall_metric.compute()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')

            # Deep copy the model if it's the best one
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_weights = model.state_dict()
                torch.save(best_model_weights, MODEL_SAVE_PATH)
                print(f'New best model saved to {MODEL_SAVE_PATH} (F1: {best_f1:.4f})')

        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F1: {best_f1:4f}')

    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model

# --- 5. Start Training (Main Execution Block) ---
if __name__ == "__main__":
    # Check if data paths are set
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("="*60)
        print("ERROR: Data directories not found. Please check paths.")
        print(f"       Checked TRAIN_DIR: {os.path.abspath(TRAIN_DIR)}")
        print(f"       Checked VAL_DIR: {os.path.abspath(VAL_DIR)}")
        print("="*60)
    else:
        print("Data directories found.")
        
        # --- 2. Create DataLoaders ---
        # THIS CODE NOW RUNS ONLY IN THE MAIN PROCESS
        print("Loading data...")
        image_datasets = {
            'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
            'val': datasets.ImageFolder(VAL_DIR, data_transforms['val'])
        }

        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
            'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print(f"Class names: {class_names}")
        print(f"Training data size: {dataset_sizes['train']}")
        print(f"Validation data size: {dataset_sizes['val']}")

        # --- 3. Initialize Model, Loss, Optimizer ---
        # THIS CODE NOW RUNS ONLY IN THE MAIN PROCESS
        print("\nInitializing model...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        model = DeeperCNN().to(device)

        # Print model summary (like your screenshot)
        print("\nModel Architecture:")
        print("="*60)
        summary(model, input_size=(3, 224, 224))
        print("="*60)

        criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print("Starting training...")
        trained_model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=NUM_EPOCHS)