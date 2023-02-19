from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from logo_maker.data_loading import ImgFolderDataset, show_image_grid

matplotlib.use("TkAgg")
EMBEDDING_DIM = 32


@dataclass
class NoiseSchedule:
    def __init__(self, beta_start: float = 0.0001, beta_end: float = 0.02, n_time_steps: int = 100):
        self.beta_schedule = torch.linspace(beta_start, beta_end, n_time_steps)
        alphas = 1 - self.beta_schedule
        alpha_cumulative_product = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_cumulative_product = torch.sqrt(alpha_cumulative_product)
        self.one_minus_sqrt_alpha_cumulative_product = torch.sqrt(1 - alpha_cumulative_product)


RANDOM_NUMBER_GENERATOR = torch.Generator()
RANDOM_NUMBER_GENERATOR.manual_seed(0)
SCHEDULE = NoiseSchedule()


def get_noisy_batch_at_step_t(
    original_batch: torch.Tensor, time_step: torch.Tensor, device="cuda", noise_schedule: NoiseSchedule = SCHEDULE
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Take in an image batch and add noise according to diffusion step `t` of schedule `noise_schedule`.

    :param original_batch: [n_img_batch, color, height, width] Batch of images to add noise to
    :param time_step: [n_img_batch] Diffusion step `t` to be applied to each image
    :param device: Device to use for calculation.
    :param noise_schedule: Schedule according to which to add noise to the images.
    :return: [n_img_batch, color, height, width] Noise used,
        [n_img_batch, color, height, width] Noised batch of images
    """

    if original_batch.shape[0] != time_step.shape[0]:
        raise ValueError(
            f"Batch size {original_batch.shape[0]} does not match number of requested diffusion steps {time_step.shape[0]}."
        )

    noise = torch.randn(size=original_batch.shape, generator=RANDOM_NUMBER_GENERATOR, device=original_batch.device)
    noisy_batch = (
            original_batch * noise_schedule.sqrt_alpha_cumulative_product[time_step][
                :, np.newaxis, np.newaxis, np.newaxis
            ]
            + noise * noise_schedule.one_minus_sqrt_alpha_cumulative_product[time_step][
                :, np.newaxis, np.newaxis, np.newaxis
            ]
    )

    return noisy_batch.to(device), noise.to(device)


class SinusoidalPositionEmbeddings(torch.nn.Module):
    """

    [1]: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    """
    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        """in: [n_time_steps] out: [n_time_steps, embedding_dimension]"""
        # see [1] for variable names
        half_d = self.embedding_dimension // 2  # d/2
        i_times_two_times_d = torch.arange(half_d, device=time_step.device) / (half_d - 1)  # i / (d/2) = 2*i/d
        n = 10000  # n
        denominator = torch.exp(math.log(n) * i_times_two_times_d)  # exp(ln(n) * 2 * i / d) = n ** (2 * i / d)
        sin_cos_arg = time_step[:, np.newaxis] / denominator[np.newaxis, :]  # k / n ** (2 * i / d)
        sin_embedding = sin_cos_arg.sin()
        cos_embedding = sin_cos_arg.cos()
        # note ordering sin/cos in colab notebook not as in [1] -> use ordering from [1] here
        sin_cos_alternating = torch.zeros((sin_cos_arg.shape[0], sin_cos_arg.shape[1] * 2), device=time_step.device)
        sin_cos_alternating[:, 0::2] = sin_embedding
        sin_cos_alternating[:, 1::2] = cos_embedding
        return sin_cos_alternating


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel: tuple[int, int],
        stride: int = 1,
        padding: int | str = 0,
        activation: torch.nn.modules.activation = torch.nn.LeakyReLU(),
        embedding_dim: int = EMBEDDING_DIM,
        do_norm: bool = True,
        do_dropout: bool = False,
        do_transpose: bool = False,
    ) -> None:
        super().__init__()
        self.time_mlp = torch.nn.Linear(embedding_dim, channels_out)

        if do_norm:
            self.norm = torch.nn.LazyBatchNorm2d()
        else:
            self.norm = torch.nn.Identity()

        if do_dropout:
            self.dropout = torch.nn.Dropout2d(p=0.5)
        else:
            self.dropout = torch.nn.Identity()

        if do_transpose:
            self.conv = torch.nn.ConvTranspose2d(channels_in, channels_out, kernel, stride=stride, padding=padding)
        else:
            self.conv = torch.nn.Conv2d(channels_in, channels_out, kernel, stride=stride, padding=padding)

        self.activation = activation

    def forward(self, x: torch.Tensor, time_step_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x += self.time_mlp(time_step_emb)[:, :, np.newaxis, np.newaxis]  # [n_batch, channels, n_height, n_width]
        x = self.dropout(x)
        return self.activation(x)


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(EMBEDDING_DIM),
            torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            torch.nn.LeakyReLU()
        )

        self.layers = torch.nn.ModuleList([  # input: 1 x 64 x 64
            ConvBlock(1, 32, (3, 3), stride=2, padding=1),  # 32 x 32 x 32
            ConvBlock(32, 64, (3, 3), stride=2, padding=1),  # 64 x 16 x 16
            ConvBlock(64, 128, (3, 3), stride=2, padding=1),  # 128 x 8 x 8
            ConvBlock(128, 256, (3, 3), stride=2, padding=1),  # 256 x 4 x 4
            ConvBlock(256, 128, (4, 4), stride=2, padding=1, do_transpose=True),  # 128 x 8 x 8
            ConvBlock(128, 64, (4, 4), stride=2, padding=1, do_transpose=True),  # 64 x 16 x 16
            ConvBlock(64, 32, (4, 4), stride=2, padding=1, do_transpose=True),  # 32 x 32 x 32
            ConvBlock(32, 1, (4, 4), stride=2, padding=1, do_transpose=True),  # 1 x 64 x 64
        ])

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(time_step)
        for layer_idx, layer in enumerate(self.layers): #
            x = layer(x, time_emb)

            # skip connection 0 -> 6
            if layer_idx == 0:
                residual_0_to_6 = x
            if layer_idx == 6:
                x += residual_0_to_6

            # skip connection 1 -> 5
            if layer_idx == 1:
                residual_1_to_5 = x
            if layer_idx == 5:
                x += residual_1_to_5

        return self.sigmoid(x)


@torch.no_grad()
def draw_sample_from_generator(
    model: Generator,
    n_diffusion_steps: int,
    batch_shape: tuple[int, ...],
    seed: int | None = None,
    do_plots: bool = False
) -> torch.Tensor:
    device = model.layers[0].conv.weight.device
    rand_generator = torch.Generator(device=device)
    if seed is not None:
        rand_generator.manual_seed(0)
    batch = torch.randn(size=batch_shape, generator=rand_generator, device=device)

    if do_plots:
        plot_batches = []

    for time_step in range(n_diffusion_steps):
        t = torch.full((batch_shape[0],), fill_value=time_step, device=device)
        noise = model(batch, t)
        batch -= noise
        if do_plots:
            plot_batches.append(batch.cpu())

    if do_plots:
        show_image_grid(torch.concatenate(plot_batches, dim=0))

    return batch


def train(device="cuda", n_epochs: int = 3, n_diffusion_steps: int = 100, batch_size: int = 64) -> None:
    dataset_location = Path(r"C:\Temp\data\logos")
    dataset = ImgFolderDataset(dataset_location)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = Generator()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()
    running_loss = 0
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}, Batches: ")):
            # pytorch expects tuple for size here:
            t = torch.randint(low=0, high=n_diffusion_steps - 1, size=(batch_size,))
            noisy_batch, noise = get_noisy_batch_at_step_t(batch, t, device=device)

            noise_pred = model(noisy_batch, t.to(device))
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        sample_shape = torch.Size((1, *batch.shape[1:]))
        _ = draw_sample_from_generator(model, n_diffusion_steps, sample_shape, do_plots=True)

        print(f"Epoch {epoch}/{n_epochs}, running loss = {running_loss}")
        running_loss = 0

    plt.show()


if __name__ == "__main__":
    train()


