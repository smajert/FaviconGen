from matplotlib import pyplot as plt
import pytest

import logo_maker.data_loading as dl
import logo_maker.params as params


def test_all_files_found(LogoDatasetLocation):
    file_loader = dl.LargeLogoDataset(LogoDatasetLocation, cache_files=False)
    assert len(file_loader) == 486377


@pytest.mark.skip(reason="should be run manually")
def test_image_grid(LogoDatasetLocation):
    lld = dl.load_logos(64, shuffle=True, n_images=None, cluster=params.ClusterNamesAeGrayscale.colorful_round)[1]
    batch = next(iter(lld))

    # mnist = dl.load_mnist(64, shuffle=True, n_images=None)[1]
    # batch = next(iter(mnist))[0]

    dl.show_image_grid(batch)
    plt.show()


