from diffusers import UNet2DModel
import torch

from favicon_gen.blocks import VarianceSchedule


class DiffusersModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        variance_schedule: VarianceSchedule,
        n_labels: int,
        layers_per_block: int,
    ) -> None:
        super().__init__()
        self.variance_schedule = variance_schedule
        self.n_labels = n_labels

        self.model_core = UNet2DModel(
            sample_size=32,
            dropout=0.0,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=(32, 64, 128, 256),
            downsample_type="resnet",
            upsample_type="resnet",
            down_block_types=(
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
            ),
        )

    @property
    def device(self) -> torch.device:
        return next(self.model_core.parameters()).device

    def forward(
        self, x: torch.Tensor, time_step: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
        # todo inocroporate labels here
        return self.model_core(x, time_step).sample
