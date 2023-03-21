from matplotlib import pyplot as plt
import numpy as np
import pytest
import random
import torch

from logo_maker.data_loading import LargeLogoDataset, show_image_grid
import logo_maker.stable_diffusion as sd


@pytest.fixture()
def LogoDataset(LogoDatasetLocation):
    return LargeLogoDataset(LogoDatasetLocation)


@pytest.fixture()
def LogoDataLoader(LogoDataset):
    return torch.utils.data.DataLoader(LogoDataset, batch_size=6, shuffle=False)


def test_noise_schedule_is_correct():
    noise_schedule = sd.NoiseSchedule(n_time_steps=5)
    beta = noise_schedule.beta_schedule
    prod_1 = noise_schedule.sqrt_alpha_cumulative_product
    prod_2 = noise_schedule.one_minus_sqrt_alpha_cumulative_product
    post = noise_schedule.posterior_variance
    sqrt_recip = noise_schedule.sqrt_reciprocal_alphas

    np.testing.assert_allclose(
        beta, np.array([1.0000e-04, 5.0750e-03, 1.0050e-02, 1.5025e-02, 2.0000e-02]), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        prod_1, np.array([0.9999, 0.9974, 0.9924, 0.9849, 0.9750]), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        prod_2, np.array([0.0100, 0.0719, 0.1232, 0.1731, 0.2222]), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        post, np.array([0.0000e+00, 9.8094e-05, 3.4275e-03, 7.6066e-03, 1.2141e-02]), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        sqrt_recip, np.array([1.0000, 1.0025, 1.0051, 1.0076, 1.0102]), rtol=0, atol=1e-4
    )


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
    torch.testing.assert_allclose(mean_of_image, 0.4413, rtol=0, atol=1e-4)
    torch.testing.assert_allclose(mean_of_noise, 4.7330e-05, rtol=0, atol=1e-4)

    do_plot = False
    if do_plot:
        show_image_grid(noisy_tensor)
        plt.show()


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
    noise_schedule = sd.NoiseSchedule(n_time_steps=20)
    sd.draw_sample_from_generator(model, 10, (4, 3, 32, 32), noise_schedule=noise_schedule, seed=0)


def test_model_runs(device: str = "cpu"):
    torch.random.manual_seed(0)
    random.seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    pseudo_time_steps = torch.randint(0, 10, size=(32,), device=device)
    model = sd.Generator().to(device)
    test_output = model(pseudo_batch, pseudo_time_steps)
    if device == "cpu":
        torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(0.0073, device=device), rtol=0, atol=1e-4)
    else:
        torch.testing.assert_allclose(torch.mean(test_output), torch.tensor(-0.0775, device=device), rtol=0, atol=1e-4)

