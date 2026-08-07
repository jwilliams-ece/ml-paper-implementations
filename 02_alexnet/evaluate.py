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


def validate(model: nn.Module, device: str, path: str, batch_size: int, num_worlers: int):

    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()

    running_val_loss = 0.0

    _, _, _, validationloader, _, _ = get_dataloader(batch_size=batch_size, num_workers=num_worlers,shuffle=False) # Keep False
    criterion = nn.CrossEntropyLoss()

    correct = 0.0
    total = 0.0

    with torch.no_grad():
        loop = tqdm(validationloader)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)


            # Accuracy tracking
            for prediction, label in zip(predictions,labels):
                predicted_class = torch.argmax(prediction)
                total += 1
                if predicted_class == label:
                    correct += 1

            running_val_loss += loss.item()

        accuracy = correct / total * 100
        total_validation_loss = running_val_loss / len(validationloader)

        print(f"Validation Loss: {total_validation_loss:.4f}")   
        print(f"Accuracy: {accuracy:.2f}%")   


def test(model: nn.Module, device: str, path: str, batch_size: int, num_worlers: int):
    
    model = model.to(device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.eval()

    running_val_loss = 0.0

    _, testloader, _, _, _, _ = get_dataloader(batch_size=128, num_workers=6,shuffle=False) # Keep False
    criterion = nn.CrossEntropyLoss()

    correct = 0.0
    total = 0

    with torch.no_grad():
        loop = tqdm(testloader)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Accuracy tracking
            for prediction, label in zip(predictions,labels):
                predicted_class = torch.argmax(prediction)
                total += 1
                if predicted_class == label:
                    correct += 1

            running_val_loss += loss.item()

        accuracy = correct / total * 100
        total_validation_loss = running_val_loss / len(testloader)
        print(f"Validation Loss: {total_validation_loss:.4f}")         
        print(f"Accuracy: {accuracy:.2f}%")         
        
if __name__ == "__main__":
    validate()