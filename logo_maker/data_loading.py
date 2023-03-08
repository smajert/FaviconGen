from pathlib import Path
import pickle
import tempfile
from typing import Any

from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils

pytorch_transforms = Any

FORWARD_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
])

BACKWARD_TRANSFORMS = transforms.Compose([
    transforms.RandomInvert(p=0.5),
    transforms.Lambda(lambda t: (t + 1) / 2),  # Undo scaling between [-1, 1]
    transforms.ToPILImage()
])


def tensor_to_image(tensor: Tensor) -> Image.Image:
    return BACKWARD_TRANSFORMS(tensor)


def show_image_grid(tensor: Tensor, save_as: Path | None = None ) -> None:
    img_grid = utils.make_grid(tensor)
    img_grid = BACKWARD_TRANSFORMS(img_grid.detach())

    ax = plt.gca()
    ax.imshow(img_grid)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.flush_events()
    if save_as is not None:
        plt.savefig(save_as)


class ImgFolderDataset(Dataset):
    def __init__(self, img_dir: Path, resize_to: tuple[int, int] = (64, 64), cache_files: bool = True) -> None:
        self.img_dir = img_dir
        self.transform = FORWARD_TRANSFORMS
        dir_content = img_dir.glob('**/*')
        self.img_file_locations = [file for file in dir_content if file.is_file() and file.suffix == ".png"]
        self.resizer = transforms.Resize(resize_to)
        self.cache_files = cache_files
        if self.cache_files:
            cache_file = tempfile.gettempdir() / Path(f"{resize_to[0]}x{resize_to[1]}_logo_maker_cache.pkl")
            if cache_file.exists():
                self.images = pickle.load(open(cache_file, "rb"))
            else:
                self.images = [self.resizer(Image.open(file_loc).convert("L")) for file_loc in self.img_file_locations]
                pickle.dump(self.images, open(cache_file, "wb"))

    def __len__(self) -> int:
        return len(self.img_file_locations)

    def __getitem__(self, idx) -> Tensor:
        if self.cache_files:
            return self.transform(self.images[idx])
        else:
            return self.transform(self.resizer(Image.open(self.img_file_locations[idx]).convert("L")))

