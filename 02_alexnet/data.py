import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2



def get_dataloader(
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ):

    transform = transforms.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomCrop(224),
        v2.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2, hue=0.05)
    ])
    print("Dataloader started")



    trainset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\train',
                                        transform=transform)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("Trainset Loaded")

    validationset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\val',
                                            transform=transform)
    validationloader = DataLoader(dataset=validationset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
    print("Validation Loaded")


    testset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\test',
                                        transform=transform)
    testloader = DataLoader(dataset=testset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
    print("test data loaded")

    print("Data loading complete")



    return trainset, trainloader, validationset, validationloader, testset, testloader 



