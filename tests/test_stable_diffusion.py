from matplotlib import pyplot as plt
import numpy as np
import pytest
import random
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
    mean_of_image = torch.mean(noisy_tensor[4, ...])
    mean_of_noise = torch.mean(noise[4, ...])
    torch.testing.assert_allclose(mean_of_image, -0.1451, rtol=0, atol=1e-4)
    torch.testing.assert_allclose(mean_of_noise, -0.0346, rtol=0, atol=1e-4)

    do_plot = False
    if do_plot:
        show_image_grid(noisy_tensor)


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

