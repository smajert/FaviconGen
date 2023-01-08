import pytest
from torch.utils.data import DataLoader

from logo_maker.data_loading import ImgFolderDataset, tensor_to_image
import logo_maker.stable_diffusion as sd

@pytest.fixture()
def LogoDataLoader(LogoDatasetLocation):
    dataset = ImgFolderDataset(LogoDatasetLocation)
    return DataLoader(dataset, batch_size=1)


def test_make_image_noisy(LogoDataLoader):
    image_tensor = next(iter(LogoDataLoader))[0, ...]
    noisy_tensor, noise = sd.get_noisy_sample_at_step_t(image_tensor, 20)
    noisy_img = tensor_to_image(noisy_tensor)
    noise_img = tensor_to_image(noise)
    noisy_img.show()
    noise_img.show()



