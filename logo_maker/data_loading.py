from pathlib import Path
import pickle
import tempfile
from typing import Any

import h5py
from matplotlib import pyplot as plt
import numpy as np
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


class LargeLogoDataset(Dataset):
    def __init__(self, hdf5_file_location: Path, cache_files: bool = True) -> None:
        self.transform = FORWARD_TRANSFORMS
        self.cache_files = cache_files
        self.images = None
        cache_file = tempfile.gettempdir() / Path(f"LargeLogoDataset.pkl")

        if self.cache_files:
            if cache_file.exists():
                self.images = pickle.load(open(cache_file, "rb"))

        if self.images is None:
            with h5py.File(hdf5_file_location) as file:
                stacked_images = file['data'][()]
                self.images = [
                    np.swapaxes(np.squeeze(arr), 0, -1)
                    for arr in np.split(stacked_images, stacked_images.shape[0], axis=0)
                ]
            if self.cache_files and not cache_file.exists():
                pickle.dump(self.images, open(cache_file, "wb"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tensor:
        return self.transform(self.images[idx])

