from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from pathlib import Path
from typing import List
from PIL import Image 
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import datasets


class ImageDataset(Dataset):
    """
    """
    def __init__(self, 
            data_path: str="data/train",
            transform: transforms=None
        ):
        self.classes = {i: v for i, v in enumerate(os.listdir(data_path))}
        self.label_id = {v: i for i, v in enumerate(os.listdir(data_path))}
        self.data_path = Path(data_path)
        self.transform = transform
        self._read_data()

    def _read_data(self):
        """
        """
        self.image_names = []
        self.labels = []
        for _data in os.listdir(self.data_path):
            sub_data = [f"{self.data_path}/{_data}/{file_name}" for file_name in os.listdir(self.data_path / _data)]
            self.image_names = self.image_names + sub_data
            self.labels = self.labels + [self.label_id[_data]] * len(sub_data)

    def __getitem__(self, index: int):
        image_path = self.image_names[index]
        label = self.labels[index]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
            label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.labels)


# define the CNN architecture
class CNNModel(nn.Module):
    def __init__(self, num_class:int=10):
        super().__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, num_class)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x


def main():
    """
    """
    # Hyper parameters
    batch_size = 32
    h, w, c = 32, 32, 3
    num_class = 10
    epochs = 20
    learning_rate = 0.01
    train_data_path = 'data/train'
    test_data_path = 'data/test'
    result_dir = 'model/cnn_model.pth'
    
    # Dataset
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # Train dataset
    train_tranforms = transforms.Compose([
        transforms.Resize(h),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_dataset = ImageDataset(
        data_path=train_data_path, 
        transform=train_tranforms
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test dataset
    test_tranforms = transforms.Compose([
        transforms.Resize(h),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_dataset = ImageDataset(
        data_path=test_data_path, 
        transform=test_tranforms
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = CNNModel(num_class)
    model = model.to(device)
    print(model)
    print(summary(model, input_size=(c, h, w)))
    # computes softmax and then the cross entropy
    criterion = nn.CrossEntropyLoss()
    # Optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training 
    for epoch in range(epochs):
        train_sum_loss = 0 
        train_sum_acc = 0
        test_sum_loss = 0 
        test_sum_acc = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            # Reset the gradient 
            optimizer.zero_grad()
            # Compute output
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            # Backpropagation            
            optimizer.step()
            train_sum_loss += loss.item()
            _, pred = torch.max(logit, 1)
            train_sum_acc += (pred==y).float().mean()
        
        # Testing on test set 
        model.eval()
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            with torch.no_grad():
                logit = model(x_test)
                loss = criterion(logit, y_test)
                test_sum_loss += loss.item()
            _, pred = torch.max(logit, 1)
            test_sum_acc += (pred==y_test).float().mean()
        print('Epoch {}: Train loss: {} -- Test loss: {} -- Train Acc: {} -- Test Acc: {}'.format(
            epoch, train_sum_loss/len(train_loader), test_sum_loss/len(test_loader),
            train_sum_acc/len(train_loader), test_sum_acc/len(test_loader)
        ))

    # Saving model 
    torch.save(model.state_dict(), result_dir)


if __name__ == '__main__':
    """
    """
    main()