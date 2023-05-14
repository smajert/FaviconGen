from enum import Enum
from pathlib import Path
import pickle
import tempfile
from typing import Any

import h5py
from matplotlib import pyplot as plt
import numpy as np
from torch import randperm, Tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, utils

import logo_maker.params as params

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


class ClusterNamesAeGrayscale(Enum):
    writing_on_black = 2
    round_on_white = 25


def show_image_grid(tensor: Tensor, save_as: Path | None = None) -> None:
    print(tensor.shape)
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
        cluster: ClusterNamesAeGrayscale | None = None,
        cluster_type: ClusterMethod = ClusterMethod.ae_grayscale
    ) -> None:
        self.transform = FORWARD_TRANSFORMS
        self.cache_files = cache_files
        self.images = None
        self.cluster = cluster

        cache_file = tempfile.gettempdir() / Path(f"LargeLogoDataset.pkl")

        if self.cache_files:
            if cache_file.exists():
                self.images = pickle.load(open(cache_file, "rb"))

        if self.images is None:
            with h5py.File(hdf5_file_location) as file:
                stacked_images = file["data"]
                if cluster_type == ClusterMethod.ae_grayscale:
                    clusters = file[f"labels/{cluster_type.name}"][()]
                else:
                    clusters = file[f"labels/resnet/{cluster_type.name}"][()]
                if self.cluster is not None:
                    stacked_images = stacked_images[:len(clusters)]
                    stacked_images = stacked_images[clusters == self.cluster.value, ...]
                else:
                    stacked_images = stacked_images[()]
                self.images = [
                    np.swapaxes(np.squeeze(arr), 0, -1)
                    for arr in np.split(stacked_images, stacked_images.shape[0], axis=0)
                ]
            if self.cache_files and not cache_file.exists():
                pickle.dump(self.images, open(cache_file, "wb"))

        if n_images is not None:
            self.images = self.images[:n_images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tensor:
        return self.transform(self.images[idx])


def load_logos(
    batch_size: int, shuffle: bool, n_images: int | None, cluster: ClusterNamesAeGrayscale | None = None
) -> tuple[int, DataLoader]:
    dataset_location = params.DATA_BASE_DIR / "LLD-icon.hdf5"
    logos = LargeLogoDataset(dataset_location, cluster=cluster, cache_files=False, n_images=n_images)
    print(len(logos))
    loader = DataLoader(logos, batch_size=batch_size, shuffle=shuffle)
    return len(logos), loader


def load_mnist(batch_size: int, shuffle: bool, n_images: int | None) -> tuple[int, DataLoader]:
    data_transforms = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    mnist = datasets.MNIST(tempfile.gettempdir() / Path("MNIST"), transform=data_transform, download=True)

    if n_images is not None:
        random_subset_sampler = SubsetRandomSampler(randperm(len(mnist))[:n_images])
        loader = DataLoader(mnist, batch_size=batch_size, sampler=random_subset_sampler)
        return n_images, loader
    else:
        loader = DataLoader(mnist, batch_size=batch_size, shuffle=shuffle)
        return len(mnist), loader

