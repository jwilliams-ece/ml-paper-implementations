import time
import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

# Logs function runtime
def function_runtime(func):
    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time() - t1
        print(f'Runtime for {func.__name__}: {t2:.3f}s')
    return wrapper

def get_mean_and_std():
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256,256)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = ImageFolder(
        r"C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=6
    )

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3))
        num_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean ** 2)

    print("Mean:", mean)
    print("Std:", std)


if __name__ == "__main__":
    get_mean_and_std()