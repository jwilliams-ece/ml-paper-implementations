# This is the training setup for the Alexnet NN

# Key ingredeients
# Optimizer - SGD with a batch size of 128, m = 0.9, decay=5e-4, lr=0.01
# Loss Function - Cross Entropy loss
# Epochs-90
import os
import time

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .implementation import AlexNet
from .data import get_dataloader

device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')   

def validate():

    model = AlexNet().to(device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.eval()

    running_val_loss = 0.0

    _, _, _, validationloader, _, _ = get_dataloader(batch_size=128, num_workers=6,shuffle=False) # Keep False
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        loop = tqdm(validationloader)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            running_val_loss += loss.item()

        total_validation_loss = running_val_loss / len(validationloader)
        print(f"Validation Loss: {total_validation_loss:.4f}")     


def test():
    
    model = AlexNet().to(device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.eval()

    running_val_loss = 0.0

    _, testloader, _, _, _, _ = get_dataloader(batch_size=128, num_workers=6,shuffle=False) # Keep False
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        loop = tqdm(testloader)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            running_val_loss += loss.item()

        total_validation_loss = running_val_loss / len(testloader)
        print(f"Validation Loss: {total_validation_loss:.4f}")         
        
if __name__ == "__main__":
    validate()