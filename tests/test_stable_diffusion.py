from matplotlib import pyplot as plt
import numpy as np
import pytest
import random
import torch

from logo_maker.data_loading import ImgFolderDataset, show_image_grid
import logo_maker.stable_diffusion as sd


@pytest.fixture()
def LogoDataset(LogoDatasetLocation):
    return ImgFolderDataset(LogoDatasetLocation)


@pytest.fixture()
def LogoDataLoader(LogoDataset):
    return torch.utils.data.DataLoader(LogoDataset, batch_size=6, shuffle=False)


def test_noise_schedule_is_correct():
    noise_schedule = sd.NoiseSchedule(n_time_steps=300)
    prod_1 = noise_schedule.sqrt_alpha_cumulative_product
    prod_2 = noise_schedule.one_minus_sqrt_alpha_cumulative_product
    np.testing.assert_allclose(np.array([prod_1[0], prod_1[-1]]), np.array([0.9999, 0.2192]), rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.array([prod_2[0], prod_2[-1]]), np.array([0.0100, 0.9757]), rtol=0, atol=1e-4)


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
    mean_of_image = torch.mean(noisy_tensor[4, ...])
    mean_of_noise = torch.mean(noise[4, ...])
    torch.testing.assert_allclose(mean_of_image, -0.1579, rtol=0, atol=1e-4)
    torch.testing.assert_allclose(mean_of_noise, -0.0346, rtol=0, atol=1e-4)

    do_plot = False
    if do_plot:
        show_image_grid(noisy_tensor)


def test_values_in_noise_and_image_seem_sensible(LogoDataset):
    data_loader = torch.utils.data.DataLoader(LogoDataset, batch_size=128, shuffle=False)
    image_batch = next(iter(data_loader))
    n_time_steps = 300
    noise_schedule = sd.NoiseSchedule(n_time_steps=n_time_steps)
    for t in range(0, n_time_steps, 30):
        time_step = torch.full((image_batch.shape[0], ), fill_value=t)
        noisy_tensor, noise = sd.get_noisy_batch_at_step_t(
            image_batch, time_step, noise_schedule=noise_schedule, device="cpu"
        )
        if t == 0:
            torch.testing.assert_allclose(torch.min(noisy_tensor), -1, atol=0.2, rtol=0)
            torch.testing.assert_allclose(torch.max(noisy_tensor),  1, atol=0.2, rtol=0)
        elif t > n_time_steps / 2:
            assert torch.min(noisy_tensor) < -2
            assert torch.max(noisy_tensor) > 2


def test_position_embeddings():
    time = torch.arange(0, 100, device="cpu")
    embedder = sd.SinusoidalPositionEmbeddings(512)
    embeddings = embedder(time)
    torch.testing.assert_allclose(torch.mean(embeddings), 0.3579, rtol=0, atol=1e-4)

    do_plot = False
    if do_plot:
        plt.figure()
        plt.pcolormesh(embeddings.T)
        plt.show()


def test_drawing_sample_from_module():
    model = sd.Generator()
    sd.draw_sample_from_generator(model, 10, (4, 1, 32, 32), seed=0)


def test_model_runs(device: str = "cuda"):
    torch.random.manual_seed(0)
    random.seed(0)
    pseudo_batch = torch.rand((32, 1, 64, 64), device=device)
    pseudo_time_steps = torch.randint(0, 10, size=(32,), device=device)
    model = sd.Generator().to(device)
    test_output = model(pseudo_batch, pseudo_time_steps)
    if device == "cpu":
        torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(0.6791, device=device), rtol=0, atol=1e-4)
    else:
        torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(0.5986, device=device), rtol=0, atol=1e-4)

