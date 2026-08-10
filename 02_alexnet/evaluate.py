# This is the training setup for the Alexnet NN

# Key ingredeients
# Optimizer - SGD with a batch size of 128, m = 0.9, decay=5e-4, lr=0.01
# Loss Function - Cross Entropy loss
# Epochs-90
import os
import time

import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from tqdm import tqdm

from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassConfusionMatrix
)


from .implementation import AlexNet
from .data import get_dataloader


def validate(model: nn.Module, device: str, path: str, batch_size: int, num_workers: int):

    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()

    num_classes = 100

    running_val_loss = 0.0
    correct = 0.0
    total = 0.0

    _, _, _, validationloader, _, _ = get_dataloader(batch_size=batch_size, num_workers=num_workers,shuffle=False) # Keep False
    criterion = nn.CrossEntropyLoss()

    f1_metrics = MulticlassF1Score(
        num_classes=num_classes,
        average=None,
    ).to(device)
    confusion_matrix = MulticlassConfusionMatrix(
        num_classes=num_classes,
    ).to(device)

    with torch.no_grad():
        loop = tqdm(validationloader)
        for image_crops, labels in loop:

            # crops: 10 tensors, each [B,C,224,224]
            image_crops = torch.stack(image_crops,dim=1)

            # [B,10,C,H,W]
            B,N,C,H,W = image_crops.shape

            # move data to GPU
            image_crops = image_crops.to(device)
            labels = labels.to(device)

            # treat all crops as idividual images
            image_crops = image_crops.view(B * N, C, H, W)

            # [B*10, num_classes]
            logits = model(image_crops)

            # [B,N, numclasses]
            logits = logits.view(B,N,-1)

            # compute average logits
            avg_logits = logits.mean(dim=1)
            
            # compute loss
            loss = criterion(avg_logits, labels)

            # Accuracy logging
            predicted_classes = avg_logits.argmax(dim=1)
            correct += (predicted_classes == labels).sum().item()
            total += B

            # weighted loss
            running_val_loss += loss.item() * B

            # F1 Score tracking
            f1_metrics.update(avg_logits, labels)

            # confusion matrix calculation
            confusion_matrix.update(avg_logits, labels)
            

    accuracy = correct / total * 100
    total_validation_loss = running_val_loss / total
    scores = f1_metrics.compute()
    conf_matrix = confusion_matrix.compute()

    print(f"Validation Loss: {total_validation_loss:.4f}")   
    print(f"Accuracy: {accuracy:.2f}%")   
    for class_idx, score in enumerate(scores):
        print(f"Class {class_idx}: F1 = {score:.4f}")

    conf_matrix = confusion_matrix.compute().cpu().numpy()

    rows = []

    for actual_class in range(num_classes):

        row = conf_matrix[actual_class].copy()

        correct = row[actual_class]
        total = row.sum()

        row[actual_class] = 0
        top_confused = row.argsort()[-3:][::-1]

        rows.append({
            "class": actual_class,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "most_confused_with": top_confused[0],
            "mistakes_to_top_class": row[top_confused[0]],
            "second_confused_with": top_confused[1],
            "third_confused_with": top_confused[2],
        })

    summary = pd.DataFrame(rows)

    # print(summary.to_string(index=False))



def test(model: nn.Module, device: str, path: str, batch_size: int, num_workers: int):
    
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()

    running_val_loss = 0.0

    _, testloader, _, _, _, _ = get_dataloader(batch_size=batch_size, num_workers=num_workers,shuffle=False) # Keep False
    criterion = nn.CrossEntropyLoss()

    correct = 0.0
    total = 0

    with torch.no_grad():
        loop = tqdm(testloader)
        for image_crops, labels in loop:
            image_crops = image_crops.to(device)
            labels = labels.to(device)
            predictions = model(image_crops)
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