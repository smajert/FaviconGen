import random

from matplotlib import pyplot as plt
import numpy as np
import pytest
import torch

from favicon_gen.diffusion import custom_model
from favicon_gen.data_loading import LargeLogoDataset, show_image_grid


@pytest.fixture()
def LogoDataset(LogoDatasetLocation):
    return LargeLogoDataset(LogoDatasetLocation, clusters=[25])


@pytest.fixture()
def LogoDataLoader(LogoDataset):
    return torch.utils.data.DataLoader(LogoDataset, batch_size=6, shuffle=False, cache_files=False)


def test_noise_schedule_is_correct():
    noise_schedule = custom_model.VarianceSchedule(beta_start_end=(0.0001, 0.02), n_time_steps=5)
    beta = noise_schedule.beta_t
    prod_1 = torch.sqrt(noise_schedule.alpha_bar_t)
    prod_2 = torch.sqrt(1 - noise_schedule.alpha_bar_t)
    post = noise_schedule.beta_tilde_t
    sqrt_recip = torch.sqrt(1 / noise_schedule.alpha_t)

    np.testing.assert_allclose(
        beta,
        np.array([1.0000e-04, 5.0750e-03, 1.0050e-02, 1.5025e-02, 2.0000e-02]),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        prod_1, np.array([0.9999, 0.9974, 0.9924, 0.9849, 0.9750]), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        prod_2, np.array([0.0100, 0.0719, 0.1232, 0.1731, 0.2222]), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        post,
        np.array([0.0000e00, 9.8094e-05, 3.4275e-03, 7.6066e-03, 1.2141e-02]),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sqrt_recip, np.array([1.0000, 1.0025, 1.0051, 1.0076, 1.0102]), rtol=0, atol=1e-4
    )


def test_make_batch_noisy(LogoDataset):
    random.seed(0)
    torch.random.manual_seed(0)
    steps_to_show = 11
    loader = torch.utils.data.DataLoader(LogoDataset, batch_size=steps_to_show, shuffle=False)
    image_batch = next(iter(loader))[0]

    n_time_steps = 100
    noise_schedule = custom_model.VarianceSchedule(
        beta_start_end=(0.0001, 0.02), n_time_steps=n_time_steps
    )
    time_steps = torch.round(torch.linspace(0, 0.99, steps=steps_to_show) * n_time_steps).to(
        torch.long
    )
    noisy_tensor, noise = custom_model.diffusion_forward_process(
        image_batch, time_steps, schedule=noise_schedule
    )
    mean_of_image = torch.mean(noisy_tensor[4, ...])
    mean_of_noise = torch.mean(noise[4, ...])
    torch.testing.assert_close(mean_of_image, torch.tensor(0.3498), rtol=0, atol=1e-4)
    torch.testing.assert_close(mean_of_noise, torch.tensor(0.0187), rtol=0, atol=1e-4)

    do_plot = False
    if do_plot:
        show_image_grid(noisy_tensor)
        plt.show()


def test_values_in_noise_and_image_seem_sensible(LogoDataset):
    data_loader = torch.utils.data.DataLoader(LogoDataset, batch_size=128, shuffle=False)
    image_batch = next(iter(data_loader))[0]
    n_time_steps = 300
    beta_start_end = (0.0001, 0.02)
    variance_schedule = custom_model.VarianceSchedule(
        n_time_steps=n_time_steps, beta_start_end=beta_start_end
    )
    for t in range(0, n_time_steps, 30):
        time_step = torch.full((image_batch.shape[0],), fill_value=t)
        noisy_tensor, _ = custom_model.diffusion_forward_process(
            image_batch, time_step, schedule=variance_schedule
        )
        if t == 0:
            torch.testing.assert_close(
                torch.min(noisy_tensor), torch.tensor(-1.0), atol=0.2, rtol=0
            )
            torch.testing.assert_close(torch.max(noisy_tensor), torch.tensor(1.0), atol=0.2, rtol=0)
        elif t > n_time_steps / 2:
            assert torch.min(noisy_tensor) < -2
            assert torch.max(noisy_tensor) > 2


def test_position_embeddings():
    time = torch.arange(0, 100, device="cpu")
    embedder = custom_model.SinusoidalPositionEmbeddings(512)
    embeddings = embedder(time)
    torch.testing.assert_close(torch.mean(embeddings), torch.tensor(0.3579), rtol=0, atol=1e-4)

    do_plot = False
    if do_plot:
        plt.figure()
        plt.pcolormesh(embeddings.T)
        plt.show()


def test_drawing_sample_from_module_runs():
    torch.random.manual_seed(0)
    random.seed(0)
    n_time_steps = 20
    variance_schedule = custom_model.VarianceSchedule(
        n_time_steps=n_time_steps, beta_start_end=(0.0001, 0.02)
    )
    model = custom_model.DiffusionModel(3, variance_schedule, 32)
    _ = custom_model.diffusion_backward_process(model, (4, 3, 32, 32), seed=0)


def test_diffusion_model_runs(device: str = "cpu"):
    torch.random.manual_seed(0)
    random.seed(0)
    pseudo_batch = torch.rand((32, 3, 32, 32), device=device)
    pseudo_time_steps = torch.randint(0, 10, size=(32,), device=device)
    model = custom_model.DiffusionModel(
        3, custom_model.VarianceSchedule(n_time_steps=1000, beta_start_end=(0.0001, 0.02)), 32
    ).to(device)
    _ = model(pseudo_batch, pseudo_time_steps)


def test_getting_device():
    model = custom_model.DiffusionModel(
        3, custom_model.VarianceSchedule(n_time_steps=10, beta_start_end=(0.0001, 0.02)), 32
    )
    assert model.device == torch.device("cpu")
