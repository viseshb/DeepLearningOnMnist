import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os

# Set device to CPU
device = torch.device("cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
# 1. Load and transform MNIST data
transform = transforms.ToTensor()
train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 2. Define the deep fully connected neural network
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Instantiate model, loss, and optimizer
model = DeepNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 5. Evaluate on training data
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"\nAccuracy on training data: {100 * correct / total:.2f}%")

# 6. Evaluate on test data
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy on test data: {100 * correct / total:.2f}%")

# 7. Visualize a **random** prediction from training data
index = random.randint(0, len(train_data) - 1)
image, label = train_data[index]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"True Label: {label}")
plt.show()

with torch.no_grad():
    pred = model(image.unsqueeze(0))
    print("Predicted Label:", torch.argmax(pred).item())

# Save the image
images_dir = os.path.join(script_dir, "images")
os.makedirs(images_dir, exist_ok=True) 
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"True Label: {label}")
plt.axis('off')  # hides the x and y axis
plt.savefig(os.path.join(images_dir,"sample_photo.png"), bbox_inches='tight')
plt.show()