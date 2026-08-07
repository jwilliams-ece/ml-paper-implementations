import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# ImageNet 100 channel statistics
mean = [0.4595, 0.4559, 0.3857]
std = [0.2517, 0.2373, 0.2526]

def get_dataloader(
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ):

    transform = transforms.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.ToDtype(torch.float32, scale=True),

        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(224),
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2, 
            hue=0.05
        ),

        v2.Normalize(
            mean,
            (1.0,1.0,1.0),
        )
    ])

    transform_val_test = transforms.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),                            # v2.TenCrop(size=(224,224), vertical_flip=False),
        v2.ToDtype(torch.float32, scale=True),

        v2.Normalize(
            mean,
            (1.0,1.0,1.0),
        )
    ])

    print("Starting DataLoader")

    trainset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\train',
                                        transform=transform)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("Trainset Loaded")

    validationset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\val',
                                            transform=transform_val_test)
    validationloader = DataLoader(dataset=validationset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
    print("Validationset Loaded")


    testset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\test',
                                        transform=transform_val_test)
    testloader = DataLoader(dataset=testset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
    print("Testset Loaded")

    print("Successfully completed DataLoader!")



    return trainset, trainloader, validationset, validationloader, testset, testloader 



