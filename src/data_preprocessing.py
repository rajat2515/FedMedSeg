import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path

class PneumoniaDataset(Dataset):
    """
    PyTorch Dataset class for loading data from our processed CSV splits.
    
    This class is designed to be imported by other training
    and evaluation notebooks.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str or Path): Path to the CSV file (train_split.csv or val_split.csv).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Ensure csv_file is a Path object for consistency
        self.csv_path = Path(csv_file)
        
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {self.csv_path}")
            print("Please run the data setup/preprocessing notebook first.")
            raise
            
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get path and label from the DataFrame
        img_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        
        try:
            # 1. Open Image
            # We convert to 'RGB' because X-rays are grayscale (1 channel) 
            # but most pre-trained CNNs (like ResNet) expect 3 channels.
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            print(f"This path came from {self.csv_path} at index {idx}")
            # Return a blank image to avoid crashing the whole batch
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            label = 0 # Assign a default label
        
        # 2. Apply Transformations
        # This will handle resizing and normalization (via ToTensor())
        if self.transform:
            img = self.transform(img)
            
        # 3. Return image tensor and label tensor
        # Label is cast to float for compatibility with BCELoss
        return img, torch.tensor(label, dtype=torch.float32)