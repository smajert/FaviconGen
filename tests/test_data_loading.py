from matplotlib import pyplot as plt
import pytest
from torch.utils.data import DataLoader

import logo_maker.data_loading as dl


def test_all_files_found(LogoDatasetLocation):
    file_loader = dl.LargeLogoDataset(LogoDatasetLocation, cache_files=False)
    assert len(file_loader) == 486377


@pytest.mark.skip(reason="should be run manually")
def test_image_grid(LogoDatasetLocation):
    file_loader = dl.LargeLogoDataset(LogoDatasetLocation, cluster=2, cache_files=False)
    data_loader = DataLoader(file_loader, batch_size=32, shuffle=True)
    batch = next(iter(data_loader))
    dl.show_image_grid(batch)
    plt.show()


