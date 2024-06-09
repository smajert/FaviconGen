"""
Denoising diffusion model similar to [1].
"""

import math
from pathlib import Path
import shutil
from typing import Any  # noqa: F401

import numpy as np
import torch
from tqdm import tqdm

from favicon_gen import params
from favicon_gen.blocks import ConvBlock, ResampleModi, VarianceSchedule
from favicon_gen.data_loading import load_data, show_image_grid
from favicon_gen.diffusion.diffuser_model import DiffusersModel


def diffusion_forward_process(
    original_batch: torch.Tensor,
    time_step: torch.Tensor,
    schedule: VarianceSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Take in an image batch and add noise according to diffusion step `t` of schedule `noise_schedule`.

    :param original_batch: [n_img_batch, color, height, width] Batch of images to add noise to
    :param time_step: [n_img_batch] Diffusion step `t` to be applied to each image
    :param schedule: Variance schedule according to which to add noise to the images.
    :return: [n_img_batch, color, height, width] Noise used,
        [n_img_batch, color, height, width] Noised batch of images
    """
    if original_batch.shape[0] != time_step.shape[0]:
        raise ValueError(
            f"Batch size {original_batch.shape[0]} does not match number of"
            f" requested diffusion steps {time_step.shape[0]}."
        )

    noise = torch.randn(size=original_batch.shape, device=original_batch.device)
    noisy_batch = (
        original_batch
        * torch.sqrt(schedule.alpha_bar_t[time_step])[:, np.newaxis, np.newaxis, np.newaxis]
        + noise
        * torch.sqrt(1 - schedule.alpha_bar_t[time_step])[:, np.newaxis, np.newaxis, np.newaxis]
    )

    return noisy_batch, noise


class SinusoidalPositionEmbeddings(torch.nn.Module):
    """
    Positional embedding for the time step via sinus/cosinus.
    See [3] for more information.

    :param embedding_dimension: Amount of values in the positional encoding matrix
    """

    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        """in: [n_time_steps] out: [n_time_steps, embedding_dimension]"""
        # see [3] for variable names
        half_d = self.embedding_dimension // 2  # d/2
        i_times_two_times_d = torch.arange(half_d, device=time_step.device) / (
            half_d - 1
        )  # i / (d/2) = 2*i/d
        n = 10000  # n
        denominator = torch.exp(
            math.log(n) * i_times_two_times_d
        )  # exp(ln(n) * 2 * i / d) = n ** (2 * i / d)
        sin_cos_arg = time_step[:, np.newaxis] / denominator[np.newaxis, :]  # k / n ** (2 * i / d)
        sin_embedding = sin_cos_arg.sin()
        cos_embedding = sin_cos_arg.cos()
        sin_cos_alternating = torch.zeros(
            (sin_cos_arg.shape[0], sin_cos_arg.shape[1] * 2), device=time_step.device
        )
        sin_cos_alternating[:, 0::2] = sin_embedding
        sin_cos_alternating[:, 1::2] = cos_embedding
        return sin_cos_alternating


class DiffusionModel(torch.nn.Module):
    """
    Diffusion model following the concept described in [1].

    :param in_channels: Amount of input channels (1 for grayscale MNIST, 3 for color LLD)
    :param variance_schedule: Schedule to control the fashion in which noise is added to the images.
    :param n_labels: Amount of different labels in the data (e.g. 10 for the 10 different digits in MNIST)
    """

    def __init__(
        self,
        in_channels: int,
        variance_schedule: VarianceSchedule,
        n_labels: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.variance_schedule = variance_schedule
        self.n_labels = n_labels

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(embedding_dim),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(),
        )

        self.label_embedding = torch.nn.Embedding(n_labels, embedding_dim)
        small = {"kernel_size": 2, "padding": 0}  # type: Any
        # fmt: off
        self.layers = torch.nn.ModuleList([                                         # input: in_channels x 32 x 32
            ConvBlock(in_channels, 32, resample_modus=ResampleModi.NO),             # 0: 32 x 32 x 32
            ConvBlock(32, 64, resample_modus=ResampleModi.DOWN),                    # 1: 64 x 16 x 16
            ConvBlock(64, 128, resample_modus=ResampleModi.DOWN, **small),          # 2: 128 x 8 x 8
            ConvBlock(128, 256, resample_modus=ResampleModi.DOWN, **small),         # 3: 256 x 4 x 4
            ConvBlock(256, 256, resample_modus=ResampleModi.DOWN_AND_UP),           # 4: 256 x 4 x 4  <-+ skip after 3
            ConvBlock(512, 128, resample_modus=ResampleModi.UP, **small),           # 5: 128 x 8 x 8  <-+ skip after 2
            ConvBlock(256, 64, resample_modus=ResampleModi.UP, **small),            # 6: 64 x 16 x 16 <-+ skip after 1
            ConvBlock(128, 32, resample_modus=ResampleModi.UP),                     # 7: 32 x 32 x 32 <-+ skip after 0
            ConvBlock(64, 32, resample_modus=ResampleModi.NO),                      # 8: 32 x 32 x 32
            torch.nn.Conv2d(32, in_channels, 5, padding=2)                          # 9: in_channels x 32 x 32
        ])
        # fmt: on

    def forward(
        self, x: torch.Tensor, time_step: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
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

    @property
    def device(self) -> torch.device:
        return self.layers[0].non_transform_layers[0].weight.device


@torch.no_grad()
def diffusion_backward_process(
    model: DiffusionModel,
    batch_shape: tuple[int, ...],
    guiding_factor: float | None,
    seed: int | None = None,
    label: int | None = None,
    save_sample_as: Path | None = None,
) -> torch.Tensor:
    """
    Reverse the noising process to generate an image starting from
    Gaussian noise

    :param model: Diffusion model to predict the noise at each time step
    :param batch_shape: Shape of the batch to generate; In particular, first dimension determines
        the amount of images generated.
    :param guiding_factor: Factor between 0 and 1 of generation with label to generation without label in
        classifier-free guidance. See [4] for more details. None means do not use guidance.
    :param seed: Seed for random number generator to make
    :param label: If given, will generate image of a specific class/cluster (e.g. generate a 5 from MNIST)
    :param save_sample_as: Save the generated images as a pdf
    :return: [*batch_shape] - batch of generated images
    """
    device = model.device
    rand_generator = torch.Generator(device=device)
    if seed is not None:
        rand_generator.manual_seed(seed)
    if label is None:
        labels = torch.randint(0, model.n_labels, (batch_shape[0],), device=device)
    else:
        labels = torch.full((batch_shape[0],), fill_value=label, device=device)
    batch = torch.randn(size=batch_shape, generator=rand_generator, device=device)
    variance_schedule = model.variance_schedule

    model.eval()
    for time_step in list(range(0, variance_schedule.n_steps))[::-1]:
        t = torch.full((batch_shape[0],), fill_value=time_step, device=device)
        beta = variance_schedule.beta_t[time_step]
        alpha = variance_schedule.alpha_t[time_step]
        alpha_bar = variance_schedule.alpha_bar_t[time_step]
        beta_tilde = variance_schedule.beta_tilde_t[time_step]

        if guiding_factor is None:
            noise_pred = model(batch, t, None)
        else:
            noise_pred = torch.lerp(model(batch, t, None), model(batch, t, labels), guiding_factor)

        batch = 1 / torch.sqrt(alpha) * (batch - beta / torch.sqrt(1 - alpha_bar) * noise_pred)
        if time_step != 0:
            noise = torch.randn_like(batch)
            batch = batch + torch.sqrt(beta_tilde) * noise
        if time_step == 0:
            batch = torch.clamp(batch, -1, 1)

        if save_sample_as is not None:
            plot_batches = []
            plot_time_steps = [
                int(0.9 * variance_schedule.n_steps),
                int(0.6 * variance_schedule.n_steps),
                int(0.3 * variance_schedule.n_steps),
                0,
                1,
                5,
                20,
                40,
            ]
            if time_step in plot_time_steps:
                plot_batches.append(batch.detach().cpu())

    if save_sample_as is not None:
        show_image_grid(torch.concatenate(plot_batches, dim=0), save_as=save_sample_as)

    model.train()
    return batch


def train(
    dataset_info: params.Dataset,
    diffusion_info: params.Diffusion,
    general_params: params.General,
    model_file: Path | None = None,
) -> np.ndarray:
    """
    Training loop for diffusion model.

    :param dataset_info: Dataset parameters
    :param diffusion_info: Model parameters
    :param model_file: If given, will start from the model saved there
    :return: Loss for each epoch
    """

    n_samples, data_loader = load_data(general_params.batch_size, dataset_info)
    model_storage_directory = params.OUTS_BASE_DIR

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True)

    beta_start_end = (diffusion_info.var_schedule_start, diffusion_info.var_schedule_end)
    schedule = VarianceSchedule(
        beta_start_end=beta_start_end,
        n_time_steps=diffusion_info.steps,
        device=general_params.device,
    )

    n_labels = dataset_info.n_classes
    match diffusion_info.architecture:
        case params.DiffusionArchitecture.CUSTOM:
            model = DiffusionModel(
                dataset_info.in_channels, schedule, n_labels, general_params.embedding_dim
            )
        case params.DiffusionArchitecture.UNET2D:
            model = DiffusersModel(dataset_info.in_channels, schedule, n_labels, 2)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.to(general_params.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=general_params.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.7,
        patience=general_params.lr_reduction_patience,
        verbose=True,
        min_lr=1e-4,
    )
    loss_fn = torch.nn.MSELoss()
    running_losses = []
    running_loss = 0
    n_epochs = general_params.epochs
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        for batch, labels in data_loader:
            labels = labels.to(general_params.device)
            if np.random.random() < 0.1:
                labels = None
            batch = batch.to(general_params.device)

            optimizer.zero_grad()
            # pytorch expects tuple for size here:
            actual_batch_size = batch.shape[0]
            t = torch.randint(low=0, high=diffusion_info.steps, size=(actual_batch_size,))
            noisy_batch, noise = diffusion_forward_process(batch, t, schedule)

            noise_pred = model(noisy_batch, t.to(general_params.device), labels)
            loss = loss_fn(noise_pred, noise)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.shape[0] / n_samples

        lr_scheduler.step(loss)
        if (epoch + 1) in [
            int(rel_plot_step * n_epochs) for rel_plot_step in [0.1, 0.25, 0.5, 0.75, 1.0]
        ]:
            sample_shape = torch.Size((1, *batch.shape[1:]))
            _ = diffusion_backward_process(
                model,
                sample_shape,
                diffusion_info.guiding_factor,
                save_sample_as=model_storage_directory / f"epoch_{epoch + 1}.png",
            )

        pbar.set_description(f"Current avg. loss: {running_loss:.3f}, Epochs")
        running_losses.append(running_loss)
        running_loss = 0

    torch.save(model.state_dict(), model_storage_directory / "model.pt")

    return running_losses
