from diffusers import UNet2DModel
import torch

from favicon_gen.blocks import VarianceSchedule


class DiffusersModel(torch.nn.Module):
    """
    Diffusion model with support from the diffusers library.

    :param in_channels: Amount of channels in input images (1 for grayscale, 3 for color)
    :param variance_schedule: Schedule describing
    :param layers_per_block: Amount of conv-layers for one convlution block before down-
        sampling.
    """

    def __init__(
        self,
        in_channels: int,
        variance_schedule: VarianceSchedule,
        layers_per_block: int = 2,
    ) -> None:
        super().__init__()
        self.variance_schedule = variance_schedule

        self.model_core = UNet2DModel(
            sample_size=32,
            dropout=0.0,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=(128, 128, 128, 256, 512),
            downsample_type="resnet",
            upsample_type="resnet",
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=4,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model_core.parameters()).device

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        return self.model_core(x, time_step, return_dict=False)[0]
