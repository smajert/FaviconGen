import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from favicon_gen.blocks import ConvBlock, ResampleModi
from favicon_gen.data_loading import load_logos, load_mnist, show_image_grid
import favicon_gen.params as params


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

    noise = torch.randn(size=original_batch.shape, device=original_batch.device)
    noisy_batch = (
        original_batch * torch.sqrt(schedule.alpha_bar_t[time_step])[
            :, np.newaxis, np.newaxis, np.newaxis
        ]
        + noise * torch.sqrt(1 - schedule.alpha_bar_t[time_step])[
            :, np.newaxis, np.newaxis, np.newaxis
        ]
    )

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
    def __init__(
        self, in_channels: int, variance_schedule: VarianceSchedule, n_labels: int
    ) -> None:
        super().__init__()
        self.variance_schedule = variance_schedule
        self.n_labels = n_labels
        embedding_dim = params.EMBEDDING_DIM

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(embedding_dim),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU()
        )

        self.label_embedding = torch.nn.Embedding(n_labels, embedding_dim)
        self.layers = torch.nn.ModuleList([                                  # input: in_channels x 32 x 32
            ConvBlock(in_channels, 32, resample_modus=ResampleModi.no),      # 0: 32 x 32 x 32
            ConvBlock(32, 64, resample_modus=ResampleModi.down),             # 1: 64 x 16 x 16
            ConvBlock(64, 128, resample_modus=ResampleModi.down),            # 2: 128 x 8 x 8
            ConvBlock(128, 256, resample_modus=ResampleModi.down),           # 3: 256 x 4 x 4
            ConvBlock(256, 256, resample_modus=ResampleModi.down_and_up),    # 4: 256 x 4 x 4  <-+ skip after 3
            ConvBlock(512, 128, resample_modus=ResampleModi.up),             # 5: 128 x 8 x 8  <-+ skip after 2
            ConvBlock(256, 64, resample_modus=ResampleModi.up),              # 6: 64 x 16 x 16 <-+ skip after 1
            ConvBlock(128, 32, resample_modus=ResampleModi.up),              # 7: 32 x 32 x 32 <-+ skip after 0
            ConvBlock(64, 32, resample_modus=ResampleModi.no),               # 8: 32 x 32 x 32
            torch.nn.Conv2d(32, in_channels, 5, padding=2)                   # 9: in_channels x 32 x 32
        ])

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        time_emb = self.time_mlp(time_step)
        if labels is not None:
            time_emb += self.label_embedding(labels)

        x_down1 = self.layers[0](x, time_emb)
        x_down2 = self.layers[1](x_down1, time_emb)
        x_down3 = self.layers[2](x_down2, time_emb)
        x_down4 = self.layers[3](x_down3, time_emb)
        x = self.layers[4](x_down4, time_emb)
        x = torch.concatenate((x, x_down4), dim=1)
        x = self.layers[5](x, time_emb)
        x = torch.concatenate((x, x_down3), dim=1)
        x = self.layers[6](x, time_emb)
        x = torch.concatenate((x, x_down2), dim=1)
        x = self.layers[7](x, time_emb)
        x = torch.concatenate((x, x_down1), dim=1)
        x = self.layers[8](x, time_emb)
        x = self.layers[9](x)
        return x


@torch.no_grad()
def draw_sample_from_generator(
    model: Generator,
    batch_shape: tuple[int, ...],
    guiding_factor: float,
    seed: int | None = None,
    label: int | None = None,
    save_sample_as: Path | None = None
) -> torch.Tensor:
    device = model.layers[0].non_transform_layers[0].weight.device
    rand_generator = torch.Generator(device=device)
    if seed is not None:
        rand_generator.manual_seed(seed)
    if label is None:
        labels = torch.randint(0, model.n_labels, (batch_shape[0],), device=device)
    else:
        labels = torch.full((batch_shape[0],), fill_value=label, device=device)
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
        noise_pred = torch.lerp(model(batch, t, None), model(batch, t, labels), guiding_factor)

        batch = 1 / torch.sqrt(alpha) * (batch - beta / torch.sqrt(1 - alpha_bar) * noise_pred)
        if time_step != 0:
            noise = torch.randn_like(batch)
            batch = batch + torch.sqrt(beta_tilde) * noise
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
    dataset_info: params.Dataset,
    diffusion_info: params.Diffusion,
    use_mnist: bool,
    model_file: Path | None = None
) -> None:
    if use_mnist:
        n_samples, data_loader = load_mnist(diffusion_info.batch_size, dataset_info.shuffle, dataset_info.n_images)
        model_storage_directory = params.OUTS_BASE_DIR / "train_diffusion_model_mnist"
        in_channels = 1
    else:
        n_samples, data_loader = load_logos(
            diffusion_info.batch_size, dataset_info.shuffle, dataset_info.n_images, clusters=dataset_info.clusters
        )
        model_storage_directory = params.OUTS_BASE_DIR / "train_diffusion_model_lld"
        in_channels = 3

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True)

    beta_start_end = (diffusion_info.var_schedule_start, diffusion_info.var_schedule_end)
    schedule = VarianceSchedule(
        beta_start_end=beta_start_end,
        n_time_steps=diffusion_info.steps,
        device=params.DEVICE
    )
    n_labels = 10 if use_mnist else 100
    model = Generator(in_channels, schedule, n_labels)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.to(params.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=diffusion_info.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=3, verbose=True, min_lr=1e-4
    )
    loss_fn = torch.nn.MSELoss()
    running_losses = []
    running_loss = 0
    n_epochs = diffusion_info.epochs_mnist if use_mnist else diffusion_info.epochs_lld
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        for batch_idx, (batch, labels) in enumerate(data_loader):
            labels = labels.to(params.DEVICE)
            if np.random.random() < 0.1:
                labels = None
            batch = batch.to(params.DEVICE)

            optimizer.zero_grad()
            # pytorch expects tuple for size here:
            actual_batch_size = batch.shape[0]
            t = torch.randint(low=0, high=diffusion_info.steps, size=(actual_batch_size,))
            noisy_batch, noise = get_noisy_batch_at_step_t(batch, t, schedule)

            noise_pred = model(noisy_batch, t.to(params.DEVICE), labels)
            loss = loss_fn(noise_pred, noise)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.shape[0] / n_samples

        lr_scheduler.step(loss)
        if (epoch + 1) in [int(rel_plot_step * n_epochs) for rel_plot_step in [0.1, 0.25, 0.5, 0.75, 1.0]]:
            sample_shape = torch.Size((1, *batch.shape[1:]))
            _ = draw_sample_from_generator(
                model,
                sample_shape,
                diffusion_info.guiding_factor,
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
    parser = argparse.ArgumentParser(description="Train the diffusion model")
    parser.add_argument(
        "--use_mnist", action="store_true", help="Whether to train on MNIST instead of the Large Logo Dataset."
    )
    args = parser.parse_args()

    train(
        dataset_info=params.Dataset(),
        diffusion_info=params.Diffusion(),
        use_mnist=args.use_mnist
    )


