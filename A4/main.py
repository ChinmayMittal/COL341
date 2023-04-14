import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

### training parameters
SHOW_SAMPLES = False
BATCH_SIZE = 4
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
PRINT_INTERVAL = 2000
SMOOTHING_FACTOR = 0.9
### statistics for the CIFAR10 dataset
cifar_mean, cifar_std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
### cifar
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=cifar_mean, std = cifar_std)]) 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False)

def imshow(img):
    npimg = img.numpy()
    npimg = npimg * np.array(cifar_std).reshape(-1,1,1) + np.array(cifar_mean).reshape(-1,1,1) ### un-normalize
    plt.imshow(np.transpose(npimg, (1, 2, 0))) ### to put channels last according to matplotlib
    plt.show()
    
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(f"Images: {images.shape}, Labels: {labels.shape}") ### B * C * H * W, B

if SHOW_SAMPLES:
    imshow(torchvision.utils.make_grid(images))
    print(" ".join([classes[j] for j in labels]))


class ConvNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding="valid")
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=1, padding="valid")
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding="valid")
        self.fc1 = nn.Linear(3*3*64, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

is_cuda = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(is_cuda)
model.to(device)

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    training_correct = 0
    for batch_idx, data in enumerate(trainloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss = (loss.item()) if batch_idx == 0 else (loss.item() * (1-SMOOTHING_FACTOR) + SMOOTHING_FACTOR*running_loss)
        training_correct += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        if( batch_idx % PRINT_INTERVAL == PRINT_INTERVAL-1 ):
            print(f"Epoch:Iter => {epoch+1}:{batch_idx+1:05} | Running Loss => {running_loss:.3f}")
        
    ### TESTING ACCURACY & LOSS
    running_loss, testing_correct = 0.0, 0
    for batch_idx, data in enumerate(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        testing_correct += torch.sum(torch.argmax(logits, dim=1) == labels).item()
    
    print(f"Epoch: {epoch+1} END | TRAINING ACCURACY: {training_correct/len(trainset)*100:.2f}% | TESTING ACCURACY: {testing_correct/len(testset)*100:.2f}%")
print("Finished Training ... ")
    


    

