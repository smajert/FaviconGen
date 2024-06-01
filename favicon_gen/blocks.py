"""
Some basic building blocks used for both the variational
autoencoder and the diffusion model.
"""

from enum import Enum, auto
from typing import Any  # noqa: F401

import numpy as np
import torch

from favicon_gen import params

GENERAL_PARAMS = params.load_config().general


class ResampleModi(Enum):
    """Possible resampling modi for convolutional block"""

    UP = auto()
    DOWN = auto()
    DOWN_AND_UP = auto()
    NO = auto()


class ConvBlock(torch.nn.Module):
    """
    Simple convolutional block. Depending on `resample_modus`, will downsample the tensor,
    upsample the tensor, both (first downsampling then upsampling) or neither.

    :param channels_in: Amount of channels before the convolution block
    :param channels_out: Amount of channels after the convolution block
    :param activation: Activation function to use (e.g. torch.nn.ReLU())
    :param resample_modus: Add or remove strided convolutions based on whether
        up- or downsampling is desired
    :param kernel_size: Kernel size of strided convolutions; ignored when not resampling
    :param padding: Padding of the strided convlutions; ignored when not resampling
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        activation: torch.nn.modules.module.Module = torch.nn.LeakyReLU(),
        resample_modus: ResampleModi = ResampleModi.NO,
        kernel_size: int = 4,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.resample_modus = resample_modus
        embedding_dimension = GENERAL_PARAMS.embedding_dim
        norm_fn = torch.nn.LazyBatchNorm2d if GENERAL_PARAMS.do_norm else torch.nn.Identity

        self.conv_in: torch.nn.Conv2d | torch.nn.Identity
        self.conv_out: torch.nn.ConvTranspose2d | torch.nn.Identity
        conv_conf = {"kernel_size": kernel_size, "padding": padding, "stride": 2}  # type: Any
        match resample_modus:
            case ResampleModi.UP:
                self.conv_in = torch.nn.Identity()
                self.conv_out = torch.nn.ConvTranspose2d(channels_in, channels_out, **conv_conf)
                channels_non_transform_conv = (
                    (channels_in, channels_in),
                    (channels_in, channels_in),
                )
            case ResampleModi.DOWN:
                self.conv_in = torch.nn.Conv2d(channels_in, channels_in, **conv_conf)
                self.conv_out = torch.nn.Identity()
                channels_non_transform_conv = (
                    (channels_in, channels_out),
                    (channels_out, channels_out),
                )
            case ResampleModi.DOWN_AND_UP:
                self.conv_in = torch.nn.Conv2d(channels_in, channels_in, **conv_conf)
                self.conv_out = torch.nn.ConvTranspose2d(2 * channels_in, channels_out, **conv_conf)
                channels_non_transform_conv = (
                    (channels_in, 2 * channels_in),
                    (2 * channels_in, 2 * channels_in),
                )
            case ResampleModi.NO:
                self.conv_in = torch.nn.Identity()
                self.conv_out = torch.nn.Identity()
                channels_non_transform_conv = (
                    (channels_in, channels_out),
                    (channels_out, channels_out),
                )

        self.time_mlp = torch.nn.Linear(embedding_dimension, channels_in)

        self.non_transform_layers = torch.nn.ModuleList()
        for chs in channels_non_transform_conv:
            self.non_transform_layers.extend(
                [
                    torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                    norm_fn(),
                    self.activation,
                ]
            )

    def forward(self, x: torch.Tensor, time_step_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.resample_modus in (ResampleModi.DOWN, ResampleModi.DOWN_AND_UP):
            x = self.activation(self.conv_in(x))

        if time_step_emb is not None:
            time_emb = self.activation(self.time_mlp(time_step_emb))[:, :, np.newaxis, np.newaxis]
            x = x + time_emb

        for layer in self.non_transform_layers:
            x = layer(x)

        if self.resample_modus in (ResampleModi.UP, ResampleModi.DOWN_AND_UP):
            x = self.activation(self.conv_out(x))

        return x
