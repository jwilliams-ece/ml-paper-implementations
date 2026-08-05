import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def get_dataloader(
        batch_size: int,
        num_workers: int,
        shuffle: False,
    ):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])



    trainset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\train',
                                        transform=transform)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    validationset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\val',
                                            transform=transform)
    validationloader = DataLoader(dataset=validationset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)


    testset = datasets.ImageFolder(root=r'C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100\test',
                                        transform=transform)
    testloader = DataLoader(dataset=testset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)



    return trainset, trainloader, validationset, validationloader, testset, testloader 



