from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from pathlib import Path
from typing import List
from PIL import Image 
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
        return img, label

    def __len__(self):
        return len(self.labels)


def main():
    """
    """
    # Hyper parameters
    batch_size = 32
    h, w, c = 224, 224, 3
    num_class = 10
    epochs = 20
    learning_rate = 0.01
    train_data_path = 'data/train'
    test_data_path = 'data/test'
    result_dir = 'model/vgg_model.pth'
    feature_extract = True 

    # Dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Train dataset
    train_tranforms = transforms.Compose([
        transforms.Resize(input_size),
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
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_dataset = ImageDataset(
        data_path=test_data_path, 
        transform=test_tranforms
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = models.vgg16(pretrained=True)
    # set_parameter_requires_grad
    if feature_extract:
        for name, param in model.named_parameters():
            param.requires_grad = False
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_class)
    model = model.to(device)
    print(model)
    print(summary(model, input_size=(c, h, w)))
    
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
                
    # computes softmax and then the cross entropy
    criterion = nn.CrossEntropyLoss()
    # Optimizer 
    optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)

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
            # Compute output
            logit = model(x)
            loss = criterion(logit, y)
            train_sum_loss += loss.item()
            _, pred = torch.max(logit, 1)
            train_sum_acc += (pred==y).float().mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluation
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