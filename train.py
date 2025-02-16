import torch
from torch.utils.data import DataLoader, random_split
from dataset import AgeDataset
from model import AgeNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time  # Add this too

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = AgeDataset('data')
print(f"Total images: {len(dataset)}")

# Split dataset
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = AgeNet().to(device)
criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_acc = 0
start_time = time.time()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for images, labels in train_pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_pbar.set_postfix({
            'loss': running_loss/len(train_loader),
            'accuracy': 100.*correct/total
        })
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.*correct/total
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
    print(f'Validation Accuracy: {accuracy:.2f}%')
    
    # Save if best model
    if accuracy > best_acc:
        print('Saving..')
        torch.save(model.state_dict(), 'best_model.pth')
        best_acc = accuracy

print("Training finished!")