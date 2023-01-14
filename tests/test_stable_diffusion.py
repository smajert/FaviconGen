import pytest
import torch

from logo_maker.data_loading import ImgFolderDataset, show_image_grid
import logo_maker.stable_diffusion as sd

@pytest.fixture()
def LogoDataLoader(LogoDatasetLocation):
    dataset = ImgFolderDataset(LogoDatasetLocation)
    return torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False)


def test_make_batch_noisy(LogoDataLoader):
    image_batch = next(iter(LogoDataLoader))
    noise_schedule = sd.NoiseSchedule(
        beta_start=0.0001,
        beta_end=0.04,
        n_time_steps=100
    )
    noisy_tensor, noise = sd.get_noisy_batch_at_step_t(
        image_batch, torch.tensor([0, 20, 40, 60, 80, 99]), noise_schedule=noise_schedule, device="cpu"
    )
    mean_of_image_4 = torch.mean(noisy_tensor[4, ...])
    mean_of_noise_4 = torch.mean(noise[4, ...])
    torch.testing.assert_allclose(mean_of_image_4, -0.1451, rtol=0, atol=1e-4)
    torch.testing.assert_allclose(mean_of_noise_4, -0.0346, rtol=0, atol=1e-4)

    do_plot = False
    if do_plot:
        show_image_grid(noisy_tensor)



