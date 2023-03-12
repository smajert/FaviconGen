from matplotlib import pyplot as plt
import PIL
import pytest
from torch.utils.data import DataLoader

import logo_maker.data_loading as dl


def test_all_files_found(LogoDatasetLocation):
    file_loader = dl.LargeLogoDataset(LogoDatasetLocation)
    assert len(file_loader) == 486377


def test_tensor_to_image(LogoDatasetLocation):
    file_loader = dl.LargeLogoDataset(LogoDatasetLocation)
    tensor = file_loader[5]
    image = dl.tensor_to_image(tensor)
    do_plot = False
    if do_plot:
        image.show()
    assert isinstance(image, PIL.Image.Image)


#@pytest.mark.skip(reason="should be run manually")
def test_image_grid(LogoDatasetLocation):
    file_loader = dl.LargeLogoDataset(LogoDatasetLocation)
    data_loader = DataLoader(file_loader, batch_size=64)
    batch = next(iter(data_loader))
    dl.show_image_grid(batch)
    plt.show()


