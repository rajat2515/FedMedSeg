import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchmetrics
from tqdm import tqdm
from pathlib import Path
import os
import sys

# --- Import Model Architectures ---

# Model 1 (Basic CNN) ki definition, jaisa 02-Model-1-Baseline.ipynb mein tha
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
        self.fc1 = nn.Linear(in_features=64 * 56 * 56, out_features=64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# Model 2 (Deeper CNN) ko uski file se import karna
# Path setup taaki 'Models' folder mil jaaye
sys.path.append(str(Path.cwd()))
try:
    from models.model2 import DeeperCNN
    print("Successfully imported DeeperCNN from Models/model2.py")
except ImportError:
    print("ERROR: Could not import DeeperCNN. Make sure 'Models/model2.py' exists.")
    sys.exit(1)


# --- Configuration ---
project_root = Path.cwd()
VAL_DIR = project_root / 'data' / 'raw' / 'chest_xray' / 'val'
MODEL1_WEIGHTS = project_root / 'models' / 'model_1_basic_cnn_best.pth'
MODEL2_WEIGHTS = project_root / 'model_2_best.pth'
BATCH_SIZE = 32
OUTPUT_IMAGE_PATH = project_root / 'comparison_plots.png'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloader, device):
    """Runs a model over the validation set and calculates all metrics."""
    model.eval()
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    
    all_labels = []
    all_preds = []

    with torch.no_grad():
        # Dataloader ke liye progress bar
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {model.__class__.__name__}"): 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Logits ko probabilities -> binary predictions (0 ya 1) mein badalna
            preds_proba = torch.sigmoid(outputs).squeeze()
            preds_binary = (preds_proba > 0.5).int()
            
            # Metrics update karna
            f1_metric.update(preds_proba, labels)
            precision_metric.update(preds_proba, labels)
            recall_metric.update(preds_proba, labels)
            accuracy_metric.update(preds_proba, labels)
            
            # Confusion matrix ke liye results store karna
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds_binary.cpu().numpy())

    metrics = {
        "Accuracy": accuracy_metric.compute().item(),
        "F1-Score": f1_metric.compute().item(),
        "Precision": precision_metric.compute().item(),
        "Recall": recall_metric.compute().item()
    }
    
    return metrics, all_labels, all_preds

def main():
    print(f"Using device: {device}")
    
    # --- Models Load Karna ---
    print("Loading models...")
    try:
        model1 = Model1_BasicCNN().to(device)
        model1.load_state_dict(torch.load(MODEL1_WEIGHTS, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model 1 weights not found at {MODEL1_WEIGHTS}")
        return
        
    try:
        model2 = DeeperCNN().to(device)
        model2.load_state_dict(torch.load(MODEL2_WEIGHTS, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model 2 weights not found at {MODEL2_WEIGHTS}")
        return
        
    print("Models loaded successfully.")

    # --- Validation Data Load Karna ---
    # Sirf resize/normalize, koi augmentation nahi
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not VAL_DIR.exists():
        print(f"ERROR: Validation directory not found at {VAL_DIR}")
        return

    val_dataset = datasets.ImageFolder(str(VAL_DIR), val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Loaded {len(val_dataset)} validation images.")
    
    # --- Evaluation Run Karna ---
    metrics_m1, y_true_m1, y_pred_m1 = evaluate_model(model1, val_loader, device)
    metrics_m2, y_true_m2, y_pred_m2 = evaluate_model(model2, val_loader, device)

    # --- Comparison Table Banana ---
    data = {"Model 1 (Basic CNN)": metrics_m1, "Model 2 (Deeper CNN)": metrics_m2}
    df = pd.DataFrame(data).T
    df = df[["F1-Score", "Accuracy", "Precision", "Recall"]] # Columns ko order mein rakhna
    
    print("\n--- Model Comparison ---")
    print(df.to_string()) # .to_string() console pe achha dikhta hai
    
    # --- Confusion Matrices Plot Karna aur Save Karna ---
    class_names = val_dataset.classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    # Model 1 Matrix
    cm1 = confusion_matrix(y_true_m1, y_pred_m1)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=class_names)
    disp1.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Model 1 (Basic CNN)')

    # Model 2 Matrix
    cm2 = confusion_matrix(y_true_m2, y_pred_m2)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=class_names)
    disp2.plot(ax=ax2, cmap='Blues')
    ax2.set_title('Model 2 (Deeper CNN)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"\nComparison plots saved to {OUTPUT_IMAGE_PATH}")

    # --- Hypothesis ka Jawab Dena ---
    f1_m1 = metrics_m1["F1-Score"]
    f1_m2 = metrics_m2["F1-Score"]
    diff = f1_m2 - f1_m1
    
    print("\n--- Hypothesis Answer ---")
    print(f"Model 1 (Basic) F1-Score:    {f1_m1:.4f}")
    print(f"Model 2 (Deeper) F1-Score:   {f1_m2:.4f}")
    print("---------------------------")
    print(f"Difference (M2 - M1):      {diff:+.4f}")

    if diff > 0.01:
        print("\nConclusion: YES. Adding depth significantly improved the F1-Score.")
    elif diff > 0:
        print("\nConclusion: MARGINALLY. Adding depth provided a small performance boost.")
    else:
        print("\nConclusion: NO. Adding depth did not improve performance.")

# Script ko run karne ke liye
if __name__ == "__main__":
    main()