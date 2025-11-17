import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchmetrics
from tqdm import tqdm
from pathlib import Path
import os
import sys
import numpy as np

# --- 1. Path Setup ---
# Path setup taaki 'src' aur 'Models' folder mil jaaye
project_root = Path.cwd()
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
print(f"Project Root: {project_root}")

# --- 2. Import Model Architectures ---

# Model 1 (Basic CNN) ki definition
class Model1_BasicCNN(nn.Module):
    def __init__(self):
        super(Model1_BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
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

# Model 2 aur 3 ko import karna
try:
    from Models.model2 import DeeperCNN
    from Models.model3 import Model3_MobileNet
    from data_preprocessing import PneumoniaDataset # SAHI validation dataset ke liye
    print("Successfully imported all model architectures and PneumoniaDataset.")
except ImportError as e:
    print(f"ERROR: {e}")
    print("Ensure 'Models/model2.py', 'Models/model3.py', and 'src/data_preprocessing.py' exist.")
    sys.exit(1)

# --- 3. Configuration ---
VAL_CSV = project_root / 'data' / 'processed' / 'val_split.csv'
MODEL1_WEIGHTS = project_root / 'models' / 'model_1_basic_cnn_best.pth'
MODEL2_WEIGHTS = project_root / 'model_2_best.pth' # Yeh root folder mein save hua tha
MODEL3_WEIGHTS = project_root / 'models' / 'model_3_mobilenet_best.pth'

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 4. Evaluation Function ---
def evaluate_model(model, dataloader, device):
    """Model ko evaluate karke metrics nikaalna."""
    model.eval()
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {model.__class__.__name__}"): 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds_proba = torch.sigmoid(outputs).squeeze()
            preds_binary = (preds_proba > 0.5).int()
            
            f1_metric.update(preds_proba, labels)
            precision_metric.update(preds_proba, labels)
            recall_metric.update(preds_proba, labels)
            accuracy_metric.update(preds_proba, labels)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds_binary.cpu().numpy())

    metrics = {
        "Accuracy": accuracy_metric.compute().item(),
        "F1-Score": f1_metric.compute().item(),
        "Precision": precision_metric.compute().item(),
        "Recall": recall_metric.compute().item()
    }
    return metrics, all_labels, all_preds

# --- 5. Main Execution ---
def main():
    print(f"Using device: {DEVICE}")
    
    # --- Models Load Karna ---
    print("Loading all 3 models...")
    try:
        model1 = Model1_BasicCNN().to(DEVICE)
        model1.load_state_dict(torch.load(MODEL1_WEIGHTS, map_location=DEVICE))
        
        model2 = DeeperCNN().to(DEVICE)
        model2.load_state_dict(torch.load(MODEL2_WEIGHTS, map_location=DEVICE))
        
        model3 = Model3_MobileNet().to(DEVICE)
        model3.load_state_dict(torch.load(MODEL3_WEIGHTS, map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"ERROR: Could not load model weights. File not found.")
        print(e)
        return
    print("Models loaded successfully.")

    # --- Validation Data Load Karna ---
    # ImageNet stats zaroori hain (Model 3 ke liye)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not VAL_CSV.exists():
        print(f"ERROR: Validation CSV not found at {VAL_CSV}")
        return
    
    val_dataset = PneumoniaDataset(csv_file=VAL_CSV, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Loaded {len(val_dataset)} validation images from 'val_split.csv'.")
    
    # --- Evaluation Run Karna ---
    metrics_m1, y_true_m1, y_pred_m1 = evaluate_model(model1, val_loader, DEVICE)
    metrics_m2, y_true_m2, y_pred_m2 = evaluate_model(model2, val_loader, DEVICE)
    metrics_m3, y_true_m3, y_pred_m3 = evaluate_model(model3, val_loader, DEVICE)

    # --- Comparison Table Banana ---
    data = {
        "Model 1 (Basic CNN)": metrics_m1,
        "Model 2 (Deeper CNN)": metrics_m2,
        "Model 3 (MobileNetV2)": metrics_m3
    }
    df = pd.DataFrame(data).T
    df = df[["F1-Score", "Accuracy", "Precision", "Recall"]]
    
    print("\n--- FINAL MODEL COMPARISON ---")
    print(df.to_string())
    df.to_csv(project_root / "final_model_comparison.csv")
    print(f"\nTable saved to 'final_model_comparison.csv'")

    # --- Visualizations ---
    print("Generating visualizations...")
    
    # Plot 1: Bar Chart F1-Scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index, y=df['F1-Score'], palette='viridis')
    plt.title('Final Model F1-Score Comparison', fontsize=16)
    plt.ylabel('F1-Score')
    plt.xlabel('Model')
    plt.ylim(0.8, 1.0) # F1-score 0.8 se 1.0 ke beech focus karo
    plt.tight_layout()
    plt.savefig(project_root / "final_f1_scores_barchart.png")
    print(f"F1-Score bar chart saved to 'final_f1_scores_barchart.png'")
    
    # Plot 2: Confusion Matrices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Confusion Matrices Comparison', fontsize=20)
    class_names = ['NORMAL', 'PNEUMONIA']
    
    cm1 = confusion_matrix(y_true_m1, y_pred_m1)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=class_names)
    disp1.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Model 1 (Basic CNN)')
    
    cm2 = confusion_matrix(y_true_m2, y_pred_m2)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=class_names)
    disp2.plot(ax=ax2, cmap='Blues')
    ax2.set_title('Model 2 (Deeper CNN)')
    
    cm3 = confusion_matrix(y_true_m3, y_pred_m3)
    disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=class_names)
    disp3.plot(ax=ax3, cmap='Blues')
    ax3.set_title('Model 3 (MobileNetV2)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Title overlap na ho
    plt.savefig(project_root / "final_confusion_matrices.png")
    print(f"Confusion matrices saved to 'final_confusion_matrices.png'")
    
    # --- Final Verdict ---
    winner = df['F1-Score'].idxmax()
    print("\n--- 🏆 FINAL VERDICT 🏆 ---")
    print(f"The 'Winning Classification Model' is: {winner}")
    print("This model will be used for Phase 2 (Federated Learning).")
    print("--- Phase 1 Complete ---")

if __name__ == "__main__":
    main()