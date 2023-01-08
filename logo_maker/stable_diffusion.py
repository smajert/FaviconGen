from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    def __init__(self, beta_start: float = 0.0001, beta_end: float = 0.02, n_time_steps: int = 100, ):
        self.beta_schedule = torch.linspace(beta_start, beta_end, n_time_steps)
        alphas = 1 - self.beta_schedule
        alpha_cumulative_product = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_cumulative_product = torch.sqrt(alpha_cumulative_product)
        self.one_minus_sqrt_alpha_cumulative_product = 1 - self.sqrt_alpha_cumulative_product


SCHEDULE = NoiseSchedule()


def get_noisy_sample_at_step_t(
    original_image, t, device="cuda", noise_schedule: NoiseSchedule = SCHEDULE
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(original_image)
    noisy_sample = (
            original_image * noise_schedule.sqrt_alpha_cumulative_product[t]
            + noise * noise_schedule.one_minus_sqrt_alpha_cumulative_product[t]
    )

    return noisy_sample.to(device), noise.to(device)


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel: tuple[int, int],
        stride: int = 1,
        padding: int | str = 0,
        activation: torch.nn.modules.activation = torch.nn.LeakyReLU(),
        do_norm: bool = True,
        do_dropout: bool = False,
        do_transpose: bool = False,
    ) -> None:
        super().__init__()
        self.activation = activation

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.activation(x)


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList([  # input: 1 x 64 x 64
            ConvBlock(1, 32, (3, 3), stride=2, padding=1),  # 32 x 32 x 32
            ConvBlock(32, 64, (3, 3), stride=2, padding=1),  # 64 x 16 x 16
            ConvBlock(64, 128, (3, 3), stride=2, padding=1),  # 128 x 8 x 8
            ConvBlock(128, 256, (3, 3), stride=2, padding=1),  # 256 x 4 x 4
            ConvBlock(256, 128, (4, 4), stride=2, padding=1, do_transpose=True),  # 128 x 8 x 8
            ConvBlock(128, 64, (4, 4), stride=2, padding=1, do_transpose=True),  # 64 x 16 x 16
            ConvBlock(64, 32, (4, 4), stride=2, padding=1, do_transpose=True),  # 32 x 32 x 32
            ConvBlock(32, 1, (4, 4), stride=2, padding=1, do_transpose=True),  # 1 x 64 x 64
            torch.nn.Sigmoid()  # 1x 256 x 256 -- 16
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            print(x.shape)

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

        return x


def train() -> None:
    model = Generator()
    test_tensor = torch.randn((1, 1, 64, 64))
    model(test_tensor)
    print(model)


if __name__ == "__main__":
    train()


