import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

t=transforms.Compose([transforms.ToTensor(),
                      transforms.Normalize((0.5),(0.5))
                      ])

#load the dataset
train_dataset=MNIST(root='./data',train=True,download=True,transform=t)
test_dataset=MNIST(root='./data',train=False,download=True,transform=t)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

class simplenet(nn.Module):
    def __init__(self):
        super(simplenet,self).__init__()
        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
    
    def forward(self,X):
        X=X.view(-1,784)
        X=self.fc1(X)
        X=F.relu(X)
        X=self.fc2(X)
        X=F.relu(X)
        X=self.fc3(X)
        return X


model=simplenet()
criterian=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)

num_epoch=10
for epoch in range(num_epoch):
    running_loss=0.0
    for image,label in train_loader:
        output=model(image)
        loss=criterian(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss+=loss.item()
    print(f'Epoch[{epoch+1}],loss:{running_loss/len(train_loader)}')


    model.eval()
    with torch.no_grad():
        correct=0.0
        total=0.0
        for image,label in test_loader:
            output=model(image)
            _,predicted=torch.max(output,1)
            correct+=(predicted==label).sum().item()
            total+=label.size(0)
        print(f'accuracy:{(correct/total)*100}')

        