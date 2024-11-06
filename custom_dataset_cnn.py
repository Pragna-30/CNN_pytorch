import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.datasets import CIFAR10,ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torchvision import models

t=transforms.Compose([transforms.ToTensor(),
                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

dataset=ImageFolder(root=r"C:\Users\student\Downloads\archive",transforms=t)

#split a dataset into train & validation set
train_ratio=0.8
train_size=int(train_ratio*len(dataset))
test_size=len(dataset)-train_size
train_dataset,test_dataset=random_split(dataset,[train_size,test_size])

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

class cnnmodel(nn.Module):
    def __init__(self):
        super(cnnmodel,self).__init__

        self.conv1=nn.Conv2d(3,32,3,stride=2)
        self.conv2=nn.Conv2d(32,64,3)
        self.conv3=nn.Conv2d(64,128,3,stride=2)
        self.fc1=nn.Linear(128*4*4,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,15)
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
