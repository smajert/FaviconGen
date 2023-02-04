import PIL
import pytest
from torch.utils.data import DataLoader

import logo_maker.data_loading as dl


def test_all_files_found(LogoDatasetLocation):
    file_loader = dl.ImgFolderDataset(LogoDatasetLocation)
    assert len(file_loader) == 17216


def test_tensor_to_image(LogoDatasetLocation):
    file_loader = dl.ImgFolderDataset(LogoDatasetLocation)
    tensor = file_loader[1000]
    image = dl.tensor_to_image(tensor)
    do_plot = False
    if do_plot:
        image.show()
    assert isinstance(image, PIL.Image.Image)


@pytest.mark.skip(reason="should be run manually")
def test_image_grid(LogoDatasetLocation):
    file_loader = dl.ImgFolderDataset(LogoDatasetLocation)
    data_loader = DataLoader(file_loader, batch_size=32)
    batch = next(iter(data_loader))
    dl.show_image_grid(batch)


