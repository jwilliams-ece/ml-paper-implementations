import time
import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from pathlib import Path
import random
import shutil

# Change this to the location of your imagenet_100 folder.
DATASET_DIR = Path(r"C:\Users\marve\Desktop\papers\02_alexnet\data\imagenet_100")

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TEST_PERCENTAGE = 0.15
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".JPEG",
    ".png",
    ".bmp",
    ".webp",
}

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

def move_train_images_to_test() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

    random_generator = random.Random(RANDOM_SEED)

    class_folders = sorted(
        folder for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        raise RuntimeError(f"No class folders found in: {TRAIN_DIR}")

    total_moved = 0

    for train_class_dir in class_folders:
        class_name = train_class_dir.name
        test_class_dir = TEST_DIR / class_name

        images = sorted(
            file
            for file in train_class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        number_to_move = int(len(images) * TEST_PERCENTAGE)

        if number_to_move == 0:
            print(
                f"{class_name}: skipped because it contains "
                f"only {len(images)} image(s)"
            )
            continue

        selected_images = random_generator.sample(
            images,
            number_to_move,
        )

        test_class_dir.mkdir(parents=True, exist_ok=True)

        moved_for_class = 0

        for source_path in selected_images:
            destination_path = test_class_dir / source_path.name

            if destination_path.exists():
                raise FileExistsError(
                    f"Destination already exists: {destination_path}"
                )

            shutil.move(str(source_path), str(destination_path))
            moved_for_class += 1

        total_moved += moved_for_class

        print(
            f"{class_name}: moved {moved_for_class} of "
            f"{len(images)} images"
        )

    print("\nSplit complete.")
    print(f"Total images moved: {total_moved}")
    print(f"Test folder: {TEST_DIR}")


if __name__ == "__main__":
    move_train_images_to_test()