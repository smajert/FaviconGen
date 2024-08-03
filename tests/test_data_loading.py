from matplotlib import pyplot as plt
import numpy as np
import pytest

from favicon_gen import params
import favicon_gen.data_loading as dlo


def test_all_files_found(LogoDatasetLocation):
    file_loader = dlo.LargeLogoDataset(LogoDatasetLocation)
    assert len(file_loader) == 486377


def test_labels_correct_when_restricting_to_cluster():
    lld = dlo.load_logos([2, 20, 50])
    labels = lld.image_labels

    assert len(lld) == 11650
    assert np.max(labels) == 2
    assert np.min(labels) == 0


def test_image_grid_and_loading():
    _, lld_loader = dlo.load_data(
        64,
        params.Dataset(
            params.AvailableDatasets.LLD, n_images=None, shuffle=True, specific_clusters=[7]
        ),
    )

    batch = next(iter(lld_loader))[0]

    dlo.show_image_grid(batch)
    show_plot = False
    if show_plot:
        plt.show()


@pytest.mark.parametrize("n_images", [1, 32])
def test_images_are_repated_for_small_image_amounts(n_images):
    dataset_conf = params.Dataset(
        params.AvailableDatasets.LLD, n_images=n_images, shuffle=True, specific_clusters=[5]
    )
    _, lld_loader = dlo.load_data(32, dataset_conf)
    dataset_conf.name = params.AvailableDatasets.MNIST
    _, mnist_loader = dlo.load_data(32, dataset_conf)

    assert len(lld_loader.dataset) == 5000
    assert len(mnist_loader.dataset) == 5000
