import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torchvision import models

t = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomRotation(10)])
train_data = CIFAR10(root='./cifar',train=True,download=True, transform=t)
val_data = CIFAR10(root='./cifar',train=False,download=True, transform=t)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(val_data, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

print(model)

in_features = model.fc.in_features
model.fc = nn.Linear(512, 15)

for param in model.parameters():
    param.requires_grad = True

criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for image, label in train_loader:
        output = model(image)
        loss = criterian(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    print(f'Epoch-{epoch + 1}/{num_epochs}, loss: {running_loss / len(train_loader)}')

    torch.save(model.state_dict(), 'best_model.pth')

    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for img, label in test_loader:
            output = model(img)
            loss = criterian(output, label)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item() 
            running_loss += loss.item()

        print(f'epoch[{epoch + 1}/{num_epochs}], loss: {running_loss / len(test_loader)}')
        print(f'accuracy: {correct / total * 100}%')

    torch.save(model.state_dict(), 'best_model.pth')

    plt.plot(running_loss)  
    #plt.show()
