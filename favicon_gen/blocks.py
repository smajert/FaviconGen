from enum import Enum, auto

import numpy as np
import torch

import favicon_gen.params as params


class ResampleModi(Enum):
    up = auto()
    down = auto()
    down_and_up = auto()
    no = auto()


class ConvBlock(torch.nn.Module):
    """
    Simple convolutional block, adding `n_non_transform_conv_layers` before
    a conv layer that reduces/increases width and height by a factor of two, depending
    on whether `do_transpose` is set or not.

    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        activation: torch.nn.modules.activation = torch.nn.LeakyReLU(),
        resample_modus: ResampleModi = ResampleModi.no,
        kernel_size: int = 4,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.resample_modus = resample_modus
        embedding_dimension = params.EMBEDDING_DIM
        norm_fn = torch.nn.LazyBatchNorm2d if params.DO_NORM else torch.nn.Identity

        conv_conf = {"kernel_size": kernel_size, "padding": padding, "stride": 2}
        match resample_modus:
            case ResampleModi.up:
                self.conv_in = torch.nn.Identity()
                self.conv_out = torch.nn.ConvTranspose2d(channels_in, channels_out, **conv_conf)
                channels_non_transform_conv = ((channels_in, channels_in), (channels_in, channels_in))
            case ResampleModi.down:
                self.conv_in = torch.nn.Conv2d(channels_in, channels_in, **conv_conf)
                self.conv_out = torch.nn.Identity()
                channels_non_transform_conv = ((channels_in, channels_out), (channels_out, channels_out))
            case ResampleModi.down_and_up:
                self.conv_in = torch.nn.Conv2d(channels_in, channels_in, **conv_conf)
                self.conv_out = torch.nn.ConvTranspose2d(2 * channels_in, channels_out, **conv_conf)
                channels_non_transform_conv = ((channels_in, 2 * channels_in), (2 * channels_in, 2 * channels_in))
            case ResampleModi.no:
                self.conv_in = torch.nn.Identity()
                self.conv_out = torch.nn.Identity()
                channels_non_transform_conv = ((channels_in, channels_out), (channels_out, channels_out))

        self.time_mlp = torch.nn.Linear(embedding_dimension, channels_in)

        self.non_transform_layers = torch.nn.ModuleList()
        for chs in channels_non_transform_conv:
            self.non_transform_layers.extend(
                [torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1), norm_fn(), self.activation]
            )

    def forward(self, x: torch.Tensor, time_step_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.resample_modus in (ResampleModi.down, ResampleModi.down_and_up):
            x = self.activation(self.conv_in(x))

        if time_step_emb is not None:
            time_emb = self.activation(self.time_mlp(time_step_emb))[:, :, np.newaxis, np.newaxis]
            x = x + time_emb

        for layer in self.non_transform_layers:
            x = layer(x)

        if self.resample_modus in (ResampleModi.up, ResampleModi.down_and_up):
            x = self.activation(self.conv_out(x))

        return x
