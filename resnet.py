import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load datasets
train_dataset = datasets.ImageFolder('Data/train', transform=transform)
val_dataset = datasets.ImageFolder('Data/valid', transform=transform)
test_dataset = datasets.ImageFolder('Data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Load ResNet50
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
model = model.to(device)

# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 6. Training loop (20 epochs)
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

# 7. Accuracy function


def evaluate_accuracy(loader, name="Set"):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"{name} Accuracy: {accuracy:.2f}%")
    return accuracy


# 8. Evaluate on validation and test sets
evaluate_accuracy(val_loader, name="Validation")
evaluate_accuracy(test_loader, name="Test")


torch.save(model.state_dict(), 'resnet50_lung_cancer.pth')
print("Model saved to resent50_lung_cancer.pth")
