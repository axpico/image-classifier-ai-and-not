import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from datasets import load_dataset
import shutil

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Download and load the dataset
print("Downloading dataset from Hugging Face...")
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
train_dataset = dataset['train']

# Create directories for organizing data
temp_data_dir = "./temp_data"
if os.path.exists(temp_data_dir):
    shutil.rmtree(temp_data_dir)

os.makedirs("./temp_data/real", exist_ok=True)
os.makedirs("./temp_data/ai", exist_ok=True)

# Prepare data
print("Preparing data...")
for i, item in enumerate(train_dataset):
    if i % 1000 == 0:
        print(f"Processing image {i}/{len(train_dataset)}")
    
    # Handle the image correctly based on its type
    if isinstance(item['image'], Image.Image):
        img = item['image']
    else:
        img = Image.fromarray(item['image'])
    
    # Convert to RGB mode before saving as JPEG
    img = img.convert('RGB')
    
    if item['label'] == 0:  # Real image
        img.save(f"./temp_data/real/img_{i}.jpg")
    else:  # AI-generated image
        img.save(f"./temp_data/ai/img_{i}.jpg")

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset class
class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['real', 'ai']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith('.jpg'):
                        self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create datasets and dataloaders
print("Creating dataloaders...")
full_dataset = ImageClassificationDataset('./temp_data', train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Dataset size: {len(full_dataset)} images")
print(f"Training set: {train_size} images")
print(f"Validation set: {val_size} images")

# Use a pre-trained model for better performance
print("Initializing model...")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
# Rimuoviamo il sigmoid dal modello
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            # Applichiamo sigmoid per la predizione
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE).view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                # Applichiamo sigmoid per la predizione
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {history["train_loss"][-1]:.4f}, '
              f'Train Acc: {history["train_acc"][-1]:.4f}, '
              f'Val Loss: {history["val_loss"][-1]:.4f}, '
              f'Val Acc: {history["val_acc"][-1]:.4f}')
    
    return history

# Train the model
print("Starting training...")
history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.savefig('training_history.png')
plt.show()

# Save the model
torch.save(model.state_dict(), 'real_vs_ai_image_classifier.pth')
print("Model saved to 'real_vs_ai_image_classifier.pth'")

# Function to predict on new images
def predict_image(image_path):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image)
        # Applichiamo sigmoid per ottenere una probabilità
        prediction = torch.sigmoid(output).item()
        confidence = prediction if prediction > 0.5 else 1 - prediction
    
    result = "AI-generated image" if prediction > 0.5 else "Real photo"
    return result, confidence

# Example usage
print("\nTo use the model for prediction, use:")
print("result, confidence = predict_image('path_to_your_image.jpg')")
