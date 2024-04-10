"""
Loading MNIST or Large Logo Dataset (LLD);
see [2] for more details on LLD.
"""

from enum import Enum
from pathlib import Path
import tempfile
import warnings

import h5py
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataset import ConcatDataset  # noqa: F401
from torchvision import datasets, transforms, utils

from favicon_gen import params

FORWARD_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]  # Scale between [-1, 1]
)

BACKWARD_TRANSFORMS = transforms.Compose(
    [transforms.Lambda(lambda t: (t + 1) / 2), transforms.ToPILImage()]  # Undo scaling between [-1, 1]
)
# For explanation of clustering methods and process see [2]
ClusterMethod = Enum("ClusterMethod", ["ae_grayscale", "rc_32", "rc_64", "rc_128"])


def show_image_grid(tensor: Tensor, save_as: Path | None = None) -> None:
    """
    Visualize a batch as a grid of single images.

    :param tensor: [n_images, n_channels, height, width] - batch to visualize
    :param save_as: If given, plot will be saved here.
    """
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
    """
    Load the Large Logo Dataset described in [2].
    You can download the LLD-icon.hdf5 file from
    here: https://data.vision.ee.ethz.ch/sagea/lld/

    :param hdf5_file_location: Location of the LLD-icon.hdf5 file
    :param n_images: Amount of images to load, default loads all available
    :param clusters: Only load images of the given clusters (see [2] for details on cluster).
        By default, all images are loaded.
    :param cluster_type: Which cluster method to use. See [2] for details.
    """

    def __init__(
        self,
        hdf5_file_location: Path,
        n_images: int | None = None,
        clusters: list[int] | None = None,
        cluster_type: ClusterMethod = ClusterMethod.ae_grayscale,
    ) -> None:
        self.transform = FORWARD_TRANSFORMS
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
            self.images = [  # pytorch needs channel dimension first
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
    """
    Convenience function for loading LLD logos.

    :param batch_size: How many images the returned `DataLoader` should provide in at once (i.e.
        as one batch)
    :param shuffle: Whether the returned `DataLoader` shuffles the data
    :param n_images: Amount of images to load, default loads all available
    :param clusters: Only load images of the given clusters (see [2] for details on cluster).
        By default, all images are loaded.
    :return: Amount of images loaded (useful when loading all images) and DataLoader for LLD
    """
    dataset_location = params.DATA_BASE_DIR / "LLD-icon.hdf5"
    logos = LargeLogoDataset(dataset_location, clusters=clusters, n_images=n_images)
    loader = DataLoader(logos, batch_size=batch_size, shuffle=shuffle)
    if n_images is None:
        print(f"Loading {len(logos)} LLD images ...")
    return len(logos), loader


def load_mnist(batch_size: int, shuffle: bool, n_images: int | None) -> tuple[int, DataLoader]:
    """
    Convenience function for loading MNIST digits

    :param batch_size: How many images the returned `DataLoader` should provide at once (i.e.
        as one batch)
    :param shuffle: Whether the returned `DataLoader` shuffles the data
    :param n_images: Amount of images to load, default loads all available
    :return: Amount of images loaded (useful when loading all images) and DataLoader for MNIST
    """
    data_transforms = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    mnist_train = datasets.MNIST(str(tempfile.gettempdir() / Path("MNIST")), transform=data_transform, download=True)
    mnist_test = datasets.MNIST(
        str(tempfile.gettempdir() / Path("MNIST")), train=False, transform=data_transform, download=True
    )
    mnist = ConcatDataset([mnist_train, mnist_test])  # type: ConcatDataset

    if n_images is not None:
        if n_images > len(mnist):
            warnings.warn(f"Requested {n_images} images, but MNIST is only {len(mnist)} images long.")
        loader = DataLoader(Subset(mnist, list(range(len(mnist)))[:n_images]), batch_size=batch_size, shuffle=shuffle)
        return n_images, loader

    print(f"Loading {len(mnist)} MNIST images ...")
    loader = DataLoader(mnist, batch_size=batch_size, shuffle=shuffle)
    return len(mnist), loader


def load_data(batch_size: int, dataset_params: params.Dataset):
    match dataset_params.name:
        case params.AvailableDatasets.MNIST:
            return load_mnist(batch_size, dataset_params.shuffle, dataset_params.n_images)
        case params.AvailableDatasets.LLD:
            return load_logos(
                batch_size, dataset_params.shuffle, dataset_params.n_images, dataset_params.specific_clusters
            )


def get_number_of_different_labels(use_mnist: bool, clusters: list[int] | None) -> int:
    """
    Get amount of different labels (e.g. 10 for the ten digits in MNIST).

    :param use_mnist: Whether MNIST is used
    :param clusters: Amount of clusters loaded from LLD
    :return: Amount of different labels
    """

    if use_mnist:
        return 10
    if clusters is not None:
        return len(clusters)

    return 100
