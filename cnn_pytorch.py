import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

t=transforms.Compose([transforms.ToTensor(),
                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


train_data=CIFAR10(root='./data',train=True,download=True,transform=t)
test_data=CIFAR10(root='./data',train=False,download=True,transform=t)

train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(test_data,batch_size=32,shuffle=False)

class cnnmodel(nn.Module):
    def __init__(self):
        super(cnnmodel,self).__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.fc1=nn.Linear(128*4*4,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.pool=nn.MaxPool2d(2,2)

    def forward(self,X):
        X=self.conv1(X)
        X=F.relu(X)
        X=self.pool(X)
        X=self.conv2(X)
        X=self.pool(F.relu(X))
        X=self.pool(F.relu(self.conv3(X)))
        X=X.view(-1,128*4*4)
        X=torch.relu(self.fc1(X))
        X=F.relu(self.fc2(X))
        X=self.fc3(X)
        return X

model=cnnmodel()
criterian=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)

#train the model
num_epochs=10
for epoch in range(num_epochs):
    running_loss=0.0
    for image,label in train_loader:
        output=model(image)
        loss=criterian(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss+=loss.item()
    print(f'Epoch-{epoch+1/num_epochs},loss:{running_loss/len(train_loader)}')

model.eval()
correct=0.0
total=0.0
with torch.no_grad():
    for img,label in test_loader:
        output=model(img)
        loss=criterian(output,label)
        _,predicted=torch.max(output,1)
        total+=label.size(0)
        running_loss+=loss.item()
    print(f'epoch[{epoch+1/num_epochs}],loss:{running_loss/len(train_loader)}')
    print(f'accuracy:{correct/total*100}')

plt.plot(loss)
plt.show()
