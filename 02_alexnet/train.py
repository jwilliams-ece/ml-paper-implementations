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


def main():

    # Init the model and the optimizer outside the training loop

    learning_rate = 0.01
    weight_decay = 5e-4
    momentum = 0.9 

    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay
                        )

    _, trainloader, _, validationloader, _, _ = get_dataloader(batch_size=128,num_workers=6,shuffle=True)

    epochs = 1
    
    def train():
        print("Starting Training")

        for epoch in range(epochs):
            running_loss = 0.0
            average_loss = 0.0
            model.train()

            loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{epochs}]")

            for i, (inputs, labels) in enumerate(loop):

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                # Update progress bar statistics inline
                running_loss += loss.item()
                average_loss = running_loss / len(trainloader)

                stats = {
                        "loss": average_loss,
                        "lr": optimizer.param_groups[0]["lr"]
                    }

                loop.set_postfix(stats)

        save_path = "model_weights.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")




    train()
    print('Finished training')



        
if __name__ == "__main__":
    main()
