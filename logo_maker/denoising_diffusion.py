from dataclasses import dataclass
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from logo_maker.blocks import ConvBlock
from logo_maker.data_loading import ClusterNamesAeGrayscale, load_logos, load_mnist, show_image_grid
import logo_maker.params as params


@dataclass
class VarianceSchedule:
    def __init__(
        self,
        beta_start_end: tuple[float, float],
        n_time_steps: int,
        device: str = "cpu"
    ) -> None:
        """
        Linear variance schedule for the Denoising Diffusion Probabilistic Model (DDPM).
        The naming scheme is the same as in [1]. In particular, `beta_t` is the
        variance of the Gaussian noise added at time step t.

        [1] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models.” arXiv, Dec. 16, 2020.
            Accessed: Mar. 20, 2023. [Online]. Available: http://arxiv.org/abs/2006.11239

        :param beta_start_end: Start and end values of the variance during the noising process.
            Defaults to values used in [1]
        :param n_time_steps: Amount of noising steps in the model.
        """
        super().__init__()
        self.n_steps = n_time_steps
        beta_start = beta_start_end[0]
        beta_end = beta_start_end[1]
        self.beta_t = torch.linspace(beta_start, beta_end, n_time_steps, device=device)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)
        alpha_bar_t_minus_1 = torch.nn.functional.pad(self.alpha_bar_t[:-1], (1, 0), value=1)
        self.beta_tilde_t = (1 - alpha_bar_t_minus_1) / (1 - self.alpha_bar_t) * self.beta_t


def get_noisy_batch_at_step_t(
    original_batch: torch.Tensor, time_step: torch.Tensor, schedule: VarianceSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Take in an image batch and add noise according to diffusion step `t` of schedule `noise_schedule`.

    :param original_batch: [n_img_batch, color, height, width] Batch of images to add noise to
    :param time_step: [n_img_batch] Diffusion step `t` to be applied to each image
    :param device: Device to use for calculation.
    :param schedule: Variance schedule according to which to add noise to the images.
    :return: [n_img_batch, color, height, width] Noise used,
        [n_img_batch, color, height, width] Noised batch of images
    """
    if original_batch.shape[0] != time_step.shape[0]:
        raise ValueError(
            f"Batch size {original_batch.shape[0]} does not match number of requested diffusion steps {time_step.shape[0]}."
        )

    noise = torch.clamp(torch.randn(size=original_batch.shape, device=original_batch.device), -5, 5)
    noisy_batch = torch.clamp((
            original_batch * torch.sqrt(schedule.alpha_bar_t[time_step])[
                :, np.newaxis, np.newaxis, np.newaxis
            ]
            + noise * torch.sqrt(1 - schedule.alpha_bar_t[time_step])[
                :, np.newaxis, np.newaxis, np.newaxis
            ]
    ), -5, 5)

    return noisy_batch, noise


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


class Generator(torch.nn.Module):
    def __init__(self, variance_schedule: VarianceSchedule, embedding_dim: int) -> None:
        super().__init__()
        in_channels = 1 if params.USE_MNIST else 3

        self.variance_schedule = variance_schedule

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(embedding_dim),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU()
        )

        self.activation = torch.nn.LeakyReLU()
        self.layers_with_emb = torch.nn.ModuleList([  # input: in_channels x 32 x 32
            ConvBlock(in_channels, 64, self.activation, time_embedding_dimension=embedding_dim),  # 64 x 16 x 16
            ConvBlock(64, 128, self.activation, time_embedding_dimension=embedding_dim),  # 128 x 8 x 8
            ConvBlock(128, 256, self.activation, time_embedding_dimension=embedding_dim),  # 256 x 4 x 4
            ConvBlock(256, 512, self.activation, time_embedding_dimension=embedding_dim),  # 512 x 2 x 2
            ConvBlock(512, 256, self.activation, time_embedding_dimension=embedding_dim, do_transpose=True),  # 256 x 4 x 4
            ConvBlock(256, 128, self.activation, time_embedding_dimension=embedding_dim, do_transpose=True),  # 128 x 8 x 8
            ConvBlock(128, 64, self.activation, time_embedding_dimension=embedding_dim, do_transpose=True),  # 64 x 16 x 16
            ConvBlock(64, 64, self.activation, time_embedding_dimension=embedding_dim, do_transpose=True),  # 64 x 32 x 32
        ])

        self.last_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(64, in_channels, 3, padding=1),
        ])

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(time_step)
        for layer_idx, layer in enumerate(self.layers_with_emb):
            x = layer(x, time_emb)

            # skip connection 0 -> 6
            if layer_idx == 0:
                residual_0_to_6 = x
            if layer_idx == 6:
                x = x + residual_0_to_6

            # skip connection 1 -> 5
            if layer_idx == 1:
                residual_1_to_5 = x
            if layer_idx == 5:
                x = x + residual_1_to_5

            # skip connection 2 -> 4
            if layer_idx == 2:
                residual_2_to_4 = x
            if layer_idx == 4:
                x = x + residual_2_to_4

        for layer in self.last_layers:
            x = layer(x)

        return torch.tensor(5., device=x.device) * torch.nn.Tanh()(x)


@torch.no_grad()
def draw_sample_from_generator(
    model: Generator,
    batch_shape: tuple[int, ...],
    seed: int | None = None,
    save_sample_as: Path | None = None
) -> torch.Tensor:
    device = model.layers_with_emb[0].non_transform_layers[0].weight.device
    rand_generator = torch.Generator(device=device)
    if seed is not None:
        rand_generator.manual_seed(seed)
    batch = torch.randn(size=batch_shape, generator=rand_generator, device=device)
    variance_schedule = model.variance_schedule

    if save_sample_as is not None:
        plot_batches = []
        plot_time_steps = [
            int(0.9 * variance_schedule.n_steps),
            int(0.6 * variance_schedule.n_steps),
            int(0.3 * variance_schedule.n_steps),
            0, 1, 5, 20, 40,
        ]

    model.eval()
    for time_step in list(range(0, variance_schedule.n_steps))[::-1]:
        t = torch.full((batch_shape[0],), fill_value=time_step, device=device)
        beta = variance_schedule.beta_t[time_step]
        alpha = variance_schedule.alpha_t[time_step]
        alpha_bar = variance_schedule.alpha_bar_t[time_step]
        beta_tilde = variance_schedule.beta_tilde_t[time_step]
        noise_pred = model(batch, t)

        batch = 1 / torch.sqrt(alpha) * (batch - beta / torch.sqrt(1 - alpha_bar) * noise_pred)
        if time_step != 0:
            noise = torch.randn_like(batch)
            batch = batch + torch.sqrt(beta_tilde) * noise
            batch = torch.clamp(batch, -5, 5)
        if time_step == 0:
            batch = torch.clamp(batch, -1, 1)

        if save_sample_as is not None:
            if time_step in plot_time_steps:
                plot_batches.append(batch.detach().cpu())

    if save_sample_as is not None:
        show_image_grid(torch.concatenate(plot_batches, dim=0), save_as=save_sample_as)

    model.train()
    return batch


def train(
    batch_size: int,
    beta_start_end: tuple[float, float],
    cluster: ClusterNamesAeGrayscale | None,
    device: str,
    embedding_dim: int,
    learning_rate: float,
    model_file: Path | None,
    n_epochs: int,
    n_diffusion_steps: int,
    n_images: int,
    shuffle_data: bool,
) -> None:
    if params.USE_MNIST:
        n_samples, data_loader = load_mnist(batch_size, shuffle_data, n_images)
    else:
        n_samples, data_loader = load_logos(batch_size, shuffle_data, n_images, cluster=cluster)
    model_storage_directory = params.OUTS_BASE_DIR / "train_diffusion_model"

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True)

    schedule = VarianceSchedule(beta_start_end, n_diffusion_steps, device=device)
    model = Generator(schedule, embedding_dim)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    running_losses = []
    running_loss = 0
    #single_batch = [next(iter(data_loader))]
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        for batch_idx, batch in enumerate(data_loader):
            if params.USE_MNIST:  # throw away labels from MNIST
                batch = batch[0]
            batch = batch.to(device)

            optimizer.zero_grad()
            # pytorch expects tuple for size here:
            actual_batch_size = batch.shape[0]
            t = torch.randint(low=0, high=n_diffusion_steps, size=(actual_batch_size,))
            noisy_batch, noise = get_noisy_batch_at_step_t(batch, t, schedule)

            noise_pred = model(noisy_batch, t.to(device))
            loss = loss_fn(noise_pred, noise)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.shape[0] / n_samples
        if (epoch + 1) in [int(rel_plot_step * n_epochs) for rel_plot_step in [0.1, 0.25, 0.5, 0.75, 1.0]]:
            sample_shape = torch.Size((1, *batch.shape[1:]))
            _ = draw_sample_from_generator(
                model,
                sample_shape,
                save_sample_as=model_storage_directory / f"epoch_{epoch + 1}.png"
            )

        pbar.set_description(f"Current avg. loss: {running_loss:.3f}, Epochs")
        running_losses.append(running_loss)
        running_loss = 0

    torch.save(model.state_dict(), model_storage_directory / "model.pt")

    with open(model_storage_directory / "loss.csv", "w") as file:
        file.write("Epoch,Loss\n")
        for epoch, loss in enumerate(running_losses):
            file.write(f"{epoch},{loss}\n")


if __name__ == "__main__":
    model_file = None
    train(
        batch_size=params.DiffusionModelParams.BATCH_SIZE,
        beta_start_end=(params.DiffusionModelParams.VAR_SCHEDULE_START, params.DiffusionModelParams.VAR_SCHEDULE_END),
        cluster=params.CLUSTER,
        device=params.DEVICE,
        embedding_dim=params.DiffusionModelParams.EMBEDDING_DIMENSION,
        learning_rate=params.DiffusionModelParams.LEARNING_RATE,
        model_file=model_file,
        n_images=params.N_IMAGES,
        n_epochs=params.DiffusionModelParams.EPOCHS,
        n_diffusion_steps=params.DiffusionModelParams.DIFFUSION_STEPS,
        shuffle_data=params.SHUFFLE_DATA
    )


