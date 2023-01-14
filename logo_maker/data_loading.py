from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torch import arange, Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils

pytorch_transforms = Any

FORWARD_TRANSFORMS = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
])

BACKWARD_TRANSFORMS = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),  # Undo scaling between [-1, 1]
    transforms.ToPILImage()
])


def tensor_to_image(tensor: Tensor) -> Image.Image:
    return BACKWARD_TRANSFORMS(tensor)


def show_image_grid(tensor: Tensor) -> None:
    img_grid = utils.make_grid(tensor)
    plt.figure()
    img_grid = BACKWARD_TRANSFORMS(img_grid.detach())
    plt.imshow(img_grid)
    plt.gca().set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


class ImgFolderDataset(Dataset):
    def __init__(self, img_dir: Path) -> None:
        self.img_dir = img_dir
        self.transform = FORWARD_TRANSFORMS
        dir_content = img_dir.glob('**/*')
        self.img_file_locations = [file for file in dir_content if file.is_file() and file.suffix == ".png"]

    def __len__(self) -> int:
        return len(self.img_file_locations)

    def __getitem__(self, idx) -> Tensor:
        return self.transform(Image.open(self.img_file_locations[idx]).convert("L"))

