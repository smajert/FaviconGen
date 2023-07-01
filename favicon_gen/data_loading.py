from enum import Enum
from pathlib import Path
import pickle
import tempfile
from typing import Any
import warnings

import h5py
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
from torchvision import datasets, transforms, utils

import favicon_gen.params as params

pytorch_transforms = Any

FORWARD_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
])

BACKWARD_TRANSFORMS = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),  # Undo scaling between [-1, 1]
    transforms.ToPILImage()
])
# For explanation of clustering methods and process see:
# [2] A. Sage, E. Agustsson, R. Timofte, and L. Van Gool,
#    “Logo Synthesis and Manipulation with Clustered Generative Adversarial Networks,”
#     in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Jun. 2018,
#     pp. 5879–5888. doi: 10.1109/CVPR.2018.00616.
ClusterMethod = Enum("ClusterMethod", ["ae_grayscale", "rc_32", "rc_64", "rc_128"])


def show_image_grid(tensor: Tensor, save_as: Path | None = None) -> None:
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
    def __init__(
        self,
        hdf5_file_location: Path,
        cache_files: bool = True,
        n_images: int | None = None,
        cluster: params.ClusterNamesAeGrayscale | None = None,
        cluster_type: ClusterMethod = ClusterMethod.ae_grayscale
    ) -> None:
        self.transform = FORWARD_TRANSFORMS
        self.cache_files = cache_files
        self.images = None
        self.images_cluster = None
        self.selected_cluster = cluster

        cache_file = tempfile.gettempdir() / Path(f"LargeLogoDataset.pkl")

        if self.cache_files:
            if cache_file.exists():
                self.images = pickle.load(open(cache_file, "rb"))

        if self.images is None:
            with h5py.File(hdf5_file_location) as file:
                stacked_images = file["data"]
                if cluster_type == ClusterMethod.ae_grayscale:
                    self.images_cluster = file[f"labels/{cluster_type.name}"][()].astype(int)
                else:
                    self.images_cluster = file[f"labels/resnet/{cluster_type.name}"][()].astype(int)
                if self.selected_cluster is not None:
                    stacked_images = stacked_images[:len(self.images_cluster)]
                    stacked_images = stacked_images[self.images_cluster == self.selected_cluster.value, ...]
                else:
                    stacked_images = stacked_images[()]
                self.images = [
                    np.swapaxes(np.squeeze(arr), 0, -1)
                    for arr in np.split(stacked_images, stacked_images.shape[0], axis=0)
                ]
            if self.cache_files and not cache_file.exists():
                pickle.dump(self.images, open(cache_file, "wb"))

        if n_images is not None:
            if n_images > len(self.images):
                warnings.warn(f"Requested {n_images} images, but LLD(-cluster) is only {len(self.images)} long.")
            self.images = self.images[:n_images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple[int, Tensor]:
        if self.selected_cluster is not None:
            return self.transform(self.images[idx]), self.selected_cluster.value
        else:
            return self.transform(self.images[idx]), self.images_cluster[idx]


def load_logos(
    batch_size: int, shuffle: bool, n_images: int | None, cluster: params.ClusterNamesAeGrayscale | None = None
) -> tuple[int, DataLoader]:
    dataset_location = params.DATA_BASE_DIR / "LLD-icon.hdf5"
    logos = LargeLogoDataset(dataset_location, cluster=cluster, cache_files=False, n_images=n_images)
    loader = DataLoader(logos, batch_size=batch_size, shuffle=shuffle)
    if n_images is None:
        print(f"Loading {len(logos)} LLD images ...")
    return len(logos), loader


def load_mnist(batch_size: int, shuffle: bool, n_images: int | None) -> tuple[int, DataLoader]:
    data_transforms = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    mnist_train = datasets.MNIST(tempfile.gettempdir() / Path("MNIST"), transform=data_transform, download=True)
    mnist_test = datasets.MNIST(
        tempfile.gettempdir() / Path("MNIST"), train=False, transform=data_transform, download=True
    )
    mnist = ConcatDataset([mnist_train, mnist_test])

    if n_images is not None:
        if n_images > len(mnist):
            warnings.warn(f"Requested {n_images} images, but MNIST is only {len(mnist)} images long.")
        loader = DataLoader(Subset(mnist, list(range(len(mnist)))[:n_images]), batch_size=batch_size, shuffle=shuffle)
        return n_images, loader
    else:
        print(f"Loading {len(mnist)} MNIST images ...")
        loader = DataLoader(mnist, batch_size=batch_size, shuffle=shuffle)
        return len(mnist), loader

