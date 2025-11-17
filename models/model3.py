import torch
import torch.nn as nn
import torchvision.models as models

class Model3_MobileNet(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        """
        Initializes the MobileNetV2 model for transfer learning.
        
        Args:
            num_classes (int): Number of output classes (1 for binary classification).
            dropout_rate (float): Dropout rate for the custom classifier head.
        """
        super(Model3_MobileNet, self).__init__()
        
        # 1. Load the pre-trained MobileNetV2 base model
        # Hum naye 'weights' API ka istemaal kar rahe hain
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # 2. Freeze all parameters in the base model
        # Taaki training ke waqt inke weights update na hon
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 3. Replace the classifier head
        # MobileNetV2 ki features 1280 dimensions ka output deti hain
        # ...
        in_features = self.base_model.classifier[1].in_features # Yeh 1280 hoga
        self.base_model.classifier = nn.Identity()
        
        # Ab hum apna custom classifier head banayenge
        self.custom_head = nn.Sequential(
            # Pooling aur Flattening HATA DIYA GAYA HAI
            # Seedha Linear layer se shuru karo
            nn.Linear(in_features, 128),  
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes) # Final output (1 logit)
        )

    def forward(self, x):
        """Forward pass"""
        # 1. Base model se features extract karo (yeh frozen hai)
        x = self.base_model(x)
        # 2. Custom head se classification karo (yeh trainable hai)
        x = self.custom_head(x)
        return x

    def unfreeze_layers(self, num_layers_to_unfreeze=5):
        """
        Unfreezes the last 'num_layers_to_unfreeze' blocks of the base model
        for fine-tuning.
        """
        print(f"Unfreezing last {num_layers_to_unfreeze} feature blocks...")
        
        # MobileNetV2 ke feature blocks 'features' attribute mein hote hain
        all_feature_blocks = list(self.base_model.features.children())
        
        # Aakhiri 'n' blocks ko unfreeze karna
        layers_to_unfreeze = all_feature_blocks[-num_layers_to_unfreeze:]
        
        for block in layers_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
        
        print(f"Total feature blocks: {len(all_feature_blocks)}. Unfrozen {len(layers_to_unfreeze)} blocks.")