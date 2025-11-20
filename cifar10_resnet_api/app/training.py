# 1️⃣ Install packages (if not already)
# !pip install torch torchvision tqdm

# 2️⃣ Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os # Import 'os' for potential environment handling

# 3️⃣ Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 4️⃣ CIFAR-10 preprocessing
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.247, 0.243, 0.261])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.247, 0.243, 0.261])
])

# NOTE: DataLoaders with num_workers > 0 MUST be inside the __main__ guard on Windows.
# We will define them inside the main block.

# 5️⃣ Model (Define globally so it's accessible to child processes if needed, but not executed repeatedly)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 6️⃣ Loss & optimizer (Define globally)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 7️⃣ Training loop wrapped in a function
def train(train_loader, test_loader):
    num_epochs = 50
    best_val_acc = 0.0

    # Main Training Loop
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0 # Initialize correct and total for TRAINING
        total = 0   # Initialize correct and total for TRAINING
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            train_bar.set_postfix(loss=running_loss/total, acc=correct/total)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0 # ⬅️ Initialize correct_val here
        total_val = 0   # ⬅️ Initialize total_val here
        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                val_bar.set_postfix(loss=val_loss/total_val, acc=correct_val/total_val)

        # Calculate Final Epoch Metrics
        train_acc = correct / total # Calculate from training phase variables
        val_loss /= total_val
        val_acc = correct_val / total_val
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet18_cifar10.pth")
            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

        scheduler.step()
    
    torch.save(model.state_dict(), "final_cifar10.pth")

# ⭐ Crucial fix for Windows multiprocessing (DataLoader) ⭐
if __name__ == '__main__':
    # DataLoaders must be initialized here when using num_workers > 0 on Windows
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Note: On some Windows/PyTorch setups, adding this can help.
    # torch.multiprocessing.freeze_support() 
    
    train(train_loader, test_loader)