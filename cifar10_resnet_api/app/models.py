# app/models.py
import torch
from torchvision import models
import torch.nn as nn
import os

# class GoogleNet:
#     def __init__(self, device="cpu"):
#         # Initialize the model (adapt for CIFAR-10)
#         self.model = models.googlenet(weights=None, aux_logits=False)  #
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)  # 10 classes for CIFAR-10
        
#         # Load your trained weights (adjust filename if different)
# # app/models.py
#         weights_path = "model/best_googlenet_cifar10.pth"  # Correct path        
#         try:
#             state_dict = torch.load(weights_path, map_location=device)
#             self.model.load_state_dict(state_dict)
#             print(f"Loaded trained weights from {weights_path}")
#         except FileNotFoundError:
#             print(f"Error: Weights file not found at {weights_path}. Using random initialization.")
#         except Exception as e:
#             print(f"Error loading weights: {e}. Using random initialization.")
        
#         self.model.to(device)
#         self.model.eval()  # Set to evaluation mode
        
#         self.class_names = [
#             'airplane', 'automobile', 'bird', 'cat', 'deer',
#             'dog', 'frog', 'horse', 'ship', 'truck'
#         ]

# app/models.py


# CIFAR-10 classes
class ResNet18CIFAR10:
    def __init__(self, model_path="model/best_cifar10.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # Load ResNet18 model
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"ResNet18 CIFAR-10 model loaded on {self.device}.")