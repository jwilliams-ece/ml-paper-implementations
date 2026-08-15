import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from .neural_net import CBOW
from .data import(
    vocab_size, 
    get_data_loader,
)

device = (
    torch.accelerator.current_accelerator() 
    if torch.accelerator.is_available() 
    else torch.device("cpu")
)


lr = 1e-3
momentum = 0.9
EMBEDDING_DIM = 25

def main():

    epochs = 10
    window_size = 5
    batch_size = 256

    # model, data, loss function, optimizer
    model = CBOW(vocab_size=vocab_size, dimension_size=EMBEDDING_DIM).to(device=device)
    dataset, dataloader = get_data_loader(window_size=window_size,batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)

    def train(epochs):
        print("Training Started")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            loop = tqdm.tqdm(iterable=dataloader, desc=f"[{epoch + 1}/{epochs}]")

            for context, target in loop:
                context = context.to(device)
                target = target.to(device)
                # zero grad -> compute predictions -> compute loss -> backprop loss -> update weights
                optimizer.zero_grad()

                logits, _ = model(context)
                loss = criterion(logits,target)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            average_loss = running_loss / len(dataloader)

            if epoch % 2 == 0:
                print(f"Epoch: {epoch:<15} Loss: {average_loss}")


    train(epochs=epochs)



if __name__ == "__main__":
    main()



