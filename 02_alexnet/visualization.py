import torch 
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from .data import get_dataloader

trainset, trainloader, validationset, validationloader, _, _ = get_dataloader(batch_size=4, num_workers=6, shuffle=True)

classes = trainset.classes

def main():

    def imshow(img):
        # Function does not currently contain unnormalization

        # Move img to cpu and clamp vals 0-1
        img = img.cpu()
        img = img.clamp(0,1)

        # Move img to np
        npimg = img.numpy()

        # Matplotlib ecpects (H,W,C)
        # Transpose dimensions from (C,H,W) -> (H,W,C)
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.xticks(range(0,224,32))
        plt.yticks(range(0,224,32))
        plt.grid(True)
        plt.show()


    images, labels = next(iter(trainloader))

    imshow(torchvision.utils.make_grid(images))


    print("Labels:", " ".join(
        classes[label.item()] for label in labels
    ))


if __name__ == "__main__":
    main()
    


