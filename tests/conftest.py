from pathlib import Path

import pytest


@pytest.fixture
def LogoDatasetLocation():
    return Path(__file__).parents[1] / "data/LLD-icon.hdf5"
