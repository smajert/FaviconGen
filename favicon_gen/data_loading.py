from enum import Enum
from pathlib import Path
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

FORWARD_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]  # Scale between [-1, 1]
)

BACKWARD_TRANSFORMS = transforms.Compose(
    [transforms.Lambda(lambda t: (t + 1) / 2), transforms.ToPILImage()]  # Undo scaling between [-1, 1]
)
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
        n_images: int | None = None,
        clusters: list[int] | None = None,
        cluster_type: ClusterMethod = ClusterMethod.ae_grayscale,
    ) -> None:
        self.transform = FORWARD_TRANSFORMS
        self.images = None
        self.image_labels = None
        self.selected_clusters = clusters

        with h5py.File(hdf5_file_location) as file:
            stacked_images = file["data"]
            if cluster_type == ClusterMethod.ae_grayscale:
                image_clusters = file[f"labels/{cluster_type.name}"][()].astype(int)
            else:
                image_clusters = file[f"labels/resnet/{cluster_type.name}"][()].astype(int)
            if self.selected_clusters is not None:
                stacked_images = stacked_images[: len(image_clusters)]
                image_in_clusters = np.isin(image_clusters, self.selected_clusters)
                stacked_images = stacked_images[image_in_clusters, ...]
                image_clusters = image_clusters[image_in_clusters]
                # change labels so that e.g. if clusters = [2, 20] labels will be [0, 1]
                cluster_to_label = {cluster: idx for idx, cluster in enumerate(set(image_clusters))}
                self.image_labels = np.array([cluster_to_label[cluster] for cluster in image_clusters])
            else:
                stacked_images = stacked_images[()]
                self.image_labels = image_clusters
            self.images = [
                np.swapaxes(np.squeeze(arr), 0, -1) for arr in np.split(stacked_images, stacked_images.shape[0], axis=0)
            ]

        if n_images is not None:
            if n_images > len(self.images):
                warnings.warn(f"Requested {n_images} images, but LLD(-cluster) is only {len(self.images)} long.")
            self.images = self.images[:n_images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple[int, Tensor]:
        return self.transform(self.images[idx]), self.image_labels[idx]


def load_logos(
    batch_size: int, shuffle: bool, n_images: int | None, clusters: list[int] | None = None
) -> tuple[int, DataLoader]:
    dataset_location = params.DATA_BASE_DIR / "LLD-icon.hdf5"
    logos = LargeLogoDataset(dataset_location, clusters=clusters, n_images=n_images)
    loader = DataLoader(logos, batch_size=batch_size, shuffle=shuffle)
    if n_images is None:
        print(f"Loading {len(logos)} LLD images ...")
    return len(logos), loader


def load_mnist(batch_size: int, shuffle: bool, n_images: int | None) -> tuple[int, DataLoader]:
    data_transforms = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
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


def get_number_of_different_labels(use_mnist: bool, clusters: list[int] | None) -> int:
    if use_mnist:
        return 10
    elif clusters is not None:
        return len(clusters)
    else:
        return 100
