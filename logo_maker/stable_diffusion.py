from dataclasses import dataclass
import math
from pathlib import Path
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from logo_maker.data_loading import LargeLogoDataset, show_image_grid

EMBEDDING_DIM = 32
# todo naming convetions see paper XXXX (0.0001, 0.02)


@dataclass
class VarianceSchedule:
    def __init__(
        self,
        beta_start_end: tuple[float, float] = (0.0001, 0.02),
        n_time_steps: int = 1000
    ) -> None:
        """
        Linear variance schedule for the Denoising Diffusion Probabilistic Model (DDPM).
        The naming scheme is the same as in [1]. In particular, `beta_t` is the
        variance of the Gaussian noise added at time step t.

        [1] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models.” arXiv, Dec. 16, 2020.
            Accessed: Mar. 20, 2023. [Online]. Available: http://arxiv.org/abs/2006.11239

        :param beta_start_end: Start and end values of the variance during the noising process.
            Defaults to values used in [1]
        :param n_time_steps: Amount of noising steps in the model. Defaults to value used in [1].
        """
        self.n_steps = n_time_steps
        beta_start = beta_start_end[0]
        beta_end = beta_start_end[1]
        self.beta_t = torch.linspace(beta_start, beta_end, n_time_steps)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)
        alpha_bar_t_minus_1 = torch.nn.functional.pad(self.alpha_bar_t[:-1], (1, 0), value=1)
        self.beta_tilde_t = (1 - alpha_bar_t_minus_1) / (1 - self.alpha_bar_t) * self.beta_t


RANDOM_NUMBER_GENERATOR = torch.Generator()
RANDOM_NUMBER_GENERATOR.manual_seed(0)


def get_noisy_batch_at_step_t(
    original_batch: torch.Tensor, time_step: torch.Tensor, noise_schedule: VarianceSchedule, device="cuda",
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
            original_batch * torch.sqrt(noise_schedule.alpha_bar_t[time_step])[
                :, np.newaxis, np.newaxis, np.newaxis
            ]
            + noise * torch.sqrt(1 - noise_schedule.alpha_bar_t[time_step])[
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
        activation: torch.nn.modules.activation = torch.nn.LeakyReLU(),
        embedding_dim: int = EMBEDDING_DIM,
        do_norm: bool = True,
        do_transpose: bool = False,
    ) -> None:
        super().__init__()
        self.time_mlp = torch.nn.Linear(embedding_dim, channels_out)

        if do_norm:
            self.norm_1 = torch.nn.LazyBatchNorm2d()
            self.norm_2 = torch.nn.LazyBatchNorm2d()
        else:
            self.norm_1 = torch.nn.Identity()
            self.norm_2 = torch.nn.Identity()

        self.conv_1 = torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)  # width/height stay same
        self.conv_2 = torch.nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)  # width/height stay same
        if do_transpose:
            self.conv_3 = torch.nn.ConvTranspose2d(
                channels_out, channels_out, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv_3 = torch.nn.Conv2d(
                channels_out, channels_out, kernel_size=4, stride=2, padding=1
            )

        self.activation = activation

    def forward(self, x: torch.Tensor, time_step_emb: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.norm_1(self.conv_1(x)))
        time_emb = self.activation(self.time_mlp(time_step_emb))[:, :, np.newaxis, np.newaxis]
        x = x + time_emb  # [n_batch, channels, n_height, n_width]
        x = self.norm_2(self.activation(self.conv_2(x)))
        return self.activation(self.conv_3(x))


class Generator(torch.nn.Module):
    def __init__(self, variance_schedule: VarianceSchedule) -> None:
        super().__init__()
        self.variance_schedule = variance_schedule

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(EMBEDDING_DIM),
            torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            torch.nn.LeakyReLU()
        )

        self.layers = torch.nn.ModuleList([  # input: 3 x 32 x 32
            ConvBlock(3, 64),  # 64 x 16 x 16
            ConvBlock(64, 128),  # 128 x 8 x 8
            ConvBlock(128, 256),  # 256 x 4 x 4
            ConvBlock(256, 512),  # 512 x 2 x 2
            ConvBlock(512, 256, do_transpose=True),  # 256 x 4 x 4
            ConvBlock(256, 128, do_transpose=True),  # 128 x 8 x 8
            ConvBlock(128, 64, do_transpose=True),  # 64 x 16 x 16
            ConvBlock(64, 64, do_transpose=True),  # 64 x 32 x 32
        ])
        self.end_conv_1 = torch.nn.Conv2d(64, 32, 1)  # 32 x 32 x 32
        self.end_conv_2 = torch.nn.Conv2d(32, 3, 1)  # 3 x 32 x 32

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(time_step)
        for layer_idx, layer in enumerate(self.layers):
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

        x = torch.nn.LeakyReLU()(self.end_conv_1(x))
        return self.end_conv_2(x)


@torch.no_grad()
def draw_sample_from_generator(
    model: Generator,
    batch_shape: tuple[int, ...],
    seed: int | None = None,
    save_sample_as: Path | None = None
) -> torch.Tensor:
    device = model.layers[0].conv_1.weight.device
    rand_generator = torch.Generator(device=device)
    if seed is not None:
        rand_generator.manual_seed(0)
    batch = torch.randn(size=batch_shape, generator=rand_generator, device=device)
    variance_schedule = model.variance_schedule

    if save_sample_as is not None:
        plot_batches = []

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
        if time_step == 0:
            batch = torch.clamp(batch, -1, 1)

        if save_sample_as is not None:
            if time_step % 10 == 0:
                plot_batches.append(batch.detach().cpu())

    if save_sample_as is not None:
        show_image_grid(torch.concatenate(plot_batches, dim=0), save_as=save_sample_as)

    model.train()
    return batch


def train(
    device="cuda",
    n_epochs: int = 300,
    n_diffusion_steps: int = 1000,
    batch_size: int = 128,
    model_file: Path | None = None
) -> None:
    dataset_location = Path(__file__).parents[1] / "data/LLD-icon.hdf5"
    dataset = LargeLogoDataset(dataset_location, cluster=3)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tempdir = Path(tempfile.mkdtemp(prefix="logo_"))

    schedule = VarianceSchedule(n_time_steps=n_diffusion_steps)
    model = Generator(schedule)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    running_losses = []
    running_loss = 0
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}, Batches: ")):
            optimizer.zero_grad()
            # pytorch expects tuple for size here:
            actual_batch_size = batch.shape[0]
            t = torch.randint(low=0, high=n_diffusion_steps, size=(actual_batch_size,))
            noisy_batch, noise = get_noisy_batch_at_step_t(batch, t, schedule, device=device)
            # print("noise:", torch.max(noise), torch.mean(noise),  torch.min(noise))

            noise_pred = model(noisy_batch, t.to(device))
            # print("noise_pred", torch.max(noise_pred), torch.mean(noise_pred), torch.min(noise_pred))
            loss = loss_fn(noise_pred, noise)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            sample_shape = torch.Size((1, *batch.shape[1:]))
            _ = draw_sample_from_generator(
                model,
                n_diffusion_steps,
                sample_shape,
                save_sample_as=tempdir / f"epoch_{epoch}.png"
            )

        print(f"Epoch {epoch}/{n_epochs}, running loss = {running_loss}")
        running_losses.append(running_loss)
        running_loss = 0

    torch.save(model.state_dict(), tempdir / "model.pt")

    plt.figure()
    plt.plot(np.array(running_losses))
    plt.grid()
    plt.xlabel("Number of epochs")
    plt.ylabel(f"Loss for batch size {batch_size}")
    plt.savefig(tempdir / "loss_curve.pdf")
    plt.show()


if __name__ == "__main__":
    model_file = None
    train(model_file=model_file)


