from pathlib import Path

import pytest

from logo_maker.data_loading import ImgFolderDataset


@pytest.fixture
def LogoDatasetLocation():
    return Path(__file__).parents[1] / "data/logos"


def test_all_files_found(LogoDatasetLocation):
    file_loader = ImgFolderDataset(LogoDatasetLocation)
    for file in file_loader:
        print(file.shape)
    assert len(file_loader) == 17217
