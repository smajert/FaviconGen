from matplotlib import pyplot as plt
import numpy as np
import pytest
import random
import torch

from logo_maker.data_loading import LargeLogoDataset, show_image_grid
import logo_maker.stable_diffusion as sd


@pytest.fixture()
def LogoDataset(LogoDatasetLocation):
    return LargeLogoDataset(LogoDatasetLocation, cluster=2, cache_files=False)


@pytest.fixture()
def LogoDataLoader(LogoDataset):
    return torch.utils.data.DataLoader(LogoDataset, batch_size=6, shuffle=False)


def test_cosine_noise_schedule_is_correct():
    n_time_steps = 1000

    s = 0.008
    t = torch.arange(1, n_time_steps + 1)
    f_t = torch.cos(0.5 * np.pi * (t / n_time_steps + s) / (1 + s)) ** 2
    noise_schedule_cosine = sd.NoiseSchedule(n_time_steps=n_time_steps)
    assert noise_schedule_cosine.beta_t[-1] == 0.999
    torch.testing.assert_allclose(noise_schedule_cosine.alpha_bar_t[:-1], f_t[:-1] / f_t[0], rtol=0, atol=1e-5)

    do_plots = True
    if do_plots:
        noise_schedule_linear = sd.NoiseSchedule(
            linear_schedule_beta_start_end=(0.0001, 0.02), n_time_steps=n_time_steps
        )
        plt.figure()
        plt.plot(noise_schedule_linear.beta_t, label="linear_schedule")
        plt.plot(noise_schedule_cosine.beta_t, label="cosine schedule")
        plt.xlabel("Time step t")
        plt.ylabel(r"$\beta_t$")
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(noise_schedule_linear.alpha_bar_t, label="linear_schedule")
        plt.plot(noise_schedule_cosine.alpha_bar_t, label="cosine schedule")
        plt.xlabel("Time step t")
        plt.ylabel(r"$\overline{\alpha}_t$")
        plt.grid()
        plt.legend()
        plt.show()


def test_linear_noise_schedule_is_correct():
    noise_schedule = sd.NoiseSchedule(linear_schedule_beta_start_end=(0.0001, 0.02), n_time_steps=5)
    beta = noise_schedule.beta_t
    prod_1 = torch.sqrt(noise_schedule.alpha_bar_t)
    prod_2 = torch.sqrt(1 - noise_schedule.alpha_bar_t)
    post = noise_schedule.beta_tilde_t
    sqrt_recip = torch.sqrt(1 / noise_schedule.alpha_t)

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


def test_make_batch_noisy(LogoDataset):
    steps_to_show = 11
    loader = torch.utils.data.DataLoader(LogoDataset, batch_size=steps_to_show, shuffle=True)
    image_batch = next(iter(loader))

    n_time_steps = 1000
    noise_schedule = sd.NoiseSchedule(
        #linear_schedule_beta_start_end=(0.0001, 0.02),
        n_time_steps=n_time_steps
    )
    time_steps = torch.round(torch.linspace(0, 0.99, steps=steps_to_show) * n_time_steps).to(torch.long)
    print(time_steps)
    noisy_tensor, noise = sd.get_noisy_batch_at_step_t(
        image_batch, time_steps, noise_schedule=noise_schedule, device="cpu"
    )
    mean_of_image = torch.mean(noisy_tensor[4, ...])
    mean_of_noise = torch.mean(noise[4, ...])
    # torch.testing.assert_allclose(mean_of_image, 0.4413, rtol=0, atol=1e-4)
    # torch.testing.assert_allclose(mean_of_noise, 4.7330e-05, rtol=0, atol=1e-4)

    do_plot = True
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
    n_time_steps = 20
    noise_schedule = sd.NoiseSchedule(n_time_steps=n_time_steps)
    sd.draw_sample_from_generator(model, n_time_steps, (4, 3, 32, 32), noise_schedule=noise_schedule, seed=0)


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

