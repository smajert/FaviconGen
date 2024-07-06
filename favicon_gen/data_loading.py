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
from torch import Tensor, transpose
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataset import ConcatDataset  # noqa: F401
from torchvision import datasets, transforms, utils

from favicon_gen import params

FORWARD_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]  # Scale between [-1, 1]
)

BACKWARD_TRANSFORMS = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.ToPILImage(),
    ]  # Undo scaling between [-1, 1]
)
# For explanation of clustering methods and process see [2]
ClusterMethod = Enum("ClusterMethod", ["ae_grayscale", "rc_32", "rc_64", "rc_128"])


def show_image_grid(tensor: Tensor, save_as: Path | None = None) -> None:
    """
    Visualize a batch as a grid of single images.

    :param tensor: [n_images, n_channels, height, width] - batch to visualize
    :param save_as: If given, plot will be saved here.
    """
    img_grid = utils.make_grid(transpose(tensor, -2, -1))
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
    :param clusters: Only load images of the given clusters (see [2] for details on cluster).
        By default, all images are loaded.
    :param cluster_type: Which cluster method to use. See [2] for details.
    """

    def __init__(
        self,
        hdf5_file_location: Path,
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
                self.image_labels = np.array(
                    [cluster_to_label[cluster] for cluster in image_clusters]
                )
            else:
                stacked_images = stacked_images[()]
                self.image_labels = image_clusters
            self.images = [  # pytorch needs channel dimension first
                np.swapaxes(np.squeeze(arr), 0, -1)
                for arr in np.split(stacked_images, stacked_images.shape[0], axis=0)
            ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple[int, Tensor]:
        return self.transform(self.images[idx]), self.image_labels[idx]


def load_logos(clusters: list[int] | None = None) -> LargeLogoDataset:
    """
    Convenience function for loading LLD logos.

    :param clusters: Only load images of the given clusters (see [2] for details on cluster).
        By default, all images are loaded.
    :return: Dataset with logos from LLD
    """
    dataset_location = params.DATA_BASE_DIR / "LLD-icon.hdf5"
    return LargeLogoDataset(dataset_location, clusters=clusters)


def load_mnist() -> Dataset:
    """
    Convenience function for loading MNIST digits
    :return MNIST dataset (both train and test images)
    """
    data_transforms = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    mnist_train = datasets.MNIST(
        str(tempfile.gettempdir() / Path("MNIST")), transform=data_transform, download=True
    )
    mnist_test = datasets.MNIST(
        str(tempfile.gettempdir() / Path("MNIST")),
        train=False,
        transform=data_transform,
        download=True,
    )
    mnist = ConcatDataset([mnist_train, mnist_test])  # type: ConcatDataset

    return mnist


def load_data(batch_size: int, dataset_params: params.Dataset) -> tuple[int, DataLoader]:
    """
    Prepare the data loader required for training.

    If no more than 32 distinct images are selected via `dataset_params.n_images`, these
    will be repeated to contain 5000 images per epoch to make the training process
    faster.

    :param batch_size: How many images from the dataset to load for each adjustment of the model
        weights.
    :param dataset_params: What dataset to load (.name), whether to shuffle the data (.shuffle),
        how many images to load (.n_images) and, for the LLD dataset, which clusters to use
        (.specific_clusters).
    :return: Amount of images loaded and `DataLoader` for the requeted `DataSet`.
    """
    match dataset_params.name:
        case params.AvailableDatasets.MNIST:
            dataset = load_mnist()
        case params.AvailableDatasets.LLD:
            dataset = load_logos(dataset_params.specific_clusters)

    n_images = dataset_params.n_images
    if n_images is not None:
        if n_images > len(dataset):
            warnings.warn(
                f"Requested {n_images} images, but dataset is only {len(dataset)} images long."
            )
        dataset = Subset(dataset, list(range(len(dataset)))[:n_images])
        if n_images <= 32:
            n_images_to_repeat_to = 5000
            print(
                f"Only {n_images} different images requested. To make training faster, the images"
                f" are repeated so that {n_images_to_repeat_to} are available per epoch."
            )
            repetitions = int(n_images_to_repeat_to / n_images + 1)
            dataset = ConcatDataset([dataset] * repetitions)  # type: ConcatDataset
            dataset = Subset(dataset, list(range(len(dataset)))[:n_images_to_repeat_to])

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=dataset_params.shuffle)
    return len(dataset), data_loader
