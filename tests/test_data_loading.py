from matplotlib import pyplot as plt
import pytest

import favicon_gen.data_loading as testee
import favicon_gen.params as params


def test_all_files_found(LogoDatasetLocation):
    file_loader = testee.LargeLogoDataset(LogoDatasetLocation, cache_files=False)
    assert len(file_loader) == 486377


def test_image_grid_and_loading(LogoDatasetLocation):
    lld = testee.load_logos(64, shuffle=True, n_images=None, cluster=params.ClusterNamesAeGrayscale.colorful_round)[1]
    batch = next(iter(lld))[0]

    mnist = testee.load_mnist(64, shuffle=True, n_images=None)[1]
    batch = next(iter(mnist))[0]

    testee.show_image_grid(batch)
    show_plot = False
    if show_plot:
        plt.show()


