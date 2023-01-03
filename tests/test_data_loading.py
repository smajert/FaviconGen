from pathlib import Path

import PIL
import pytest

import logo_maker.data_loading as dl


@pytest.fixture
def LogoDatasetLocation():
    return Path(__file__).parents[1] / "data/logos"


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
