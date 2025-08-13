#!/usr/bin/env python3
"""
Minimal burn classification training script for CPU
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import yaml

class SimpleBurnDataset(Dataset):
    def __init__(self, image_dir, label_dir, split_file):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        # Read split file
        with open(split_file, 'r') as f:
            self.samples = [line.strip().split() for line in f.readlines()]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        
        # Load label
        label = Image.open(label_path)
        label = torch.from_numpy(np.array(label)).long()
        
        # Resize to fixed size for simplicity
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0).squeeze(0).long()
        
        return img, label

class SimpleBurnModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
        
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x

def train_burn_classification():
    print("=== Starting Burn Classification Training (CPU) ===\n")
    
    # Check if data exists
    if not os.path.exists('splits/burn/train.txt'):
        print("❌ Training split file not found. Please run the setup first.")
        return
    
    # Create datasets
    print("Loading datasets...")
    trainset = SimpleBurnDataset('image_total', 'label_total', 'splits/burn/train.txt')
    valset = SimpleBurnDataset('image_total', 'label_total', 'splits/burn/val.txt')
    
    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating model...")
    model = SimpleBurnModel(num_classes=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 10  # Reduced for testing
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for i, (images, labels) in enumerate(trainloader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}][{i}/{len(trainloader)}] Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}')
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in valloader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.numel()
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Validation Accuracy: {accuracy:.2f}%')
    
    # Save model
    os.makedirs('exp/burn/simple', exist_ok=True)
    torch.save(model.state_dict(), 'exp/burn/simple/burn_model.pth')
    print(f"\n✅ Training complete! Model saved to exp/burn/simple/burn_model.pth")
    print(f"Model can now classify burn images into: {['background', 'healthy', 'burn']}")

if __name__ == '__main__':
    train_burn_classification() 