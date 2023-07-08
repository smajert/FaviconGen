import numpy as np
import torch

import favicon_gen.params as params


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
        activation: torch.nn.modules.activation,
        kernel: int | tuple[int, int] = 4,
        stride: int = 2,
        padding: int = 1,
        n_non_transform_conv_layers: int = 2,
        time_embedding_dimension: int | None = None,
        do_norm: bool = params.DO_NORM,
        do_transpose: bool = False,
    ) -> None:
        super().__init__()
        if do_norm:
            norm_fn = torch.nn.LazyBatchNorm2d
        else:
            norm_fn = torch.nn.Identity

        self.activation = activation

        if do_transpose:
            self.conv_transform = torch.nn.ConvTranspose2d(
                channels_in, channels_out, kernel_size=kernel, stride=stride, padding=padding
            )
        else:
            self.conv_transform = torch.nn.Conv2d(
                channels_in, channels_out, kernel_size=kernel, stride=stride, padding=padding
            )

        self.time_embedding_dimension = time_embedding_dimension
        if time_embedding_dimension is not None:
            self.time_mlp = torch.nn.Linear(self.time_embedding_dimension, channels_out)

        self.non_transform_layers = torch.nn.ModuleList()
        for _ in range(n_non_transform_conv_layers):
            self.non_transform_layers.extend([
                torch.nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
                norm_fn(),
                self.activation
            ])

    def forward(self, x: torch.Tensor, time_step_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = self.activation(self.conv_transform(x))

        if time_step_emb is not None:
            if self.time_embedding_dimension is None:
                raise ValueError("Time step given, but no embedding dimension specified")
            time_emb = self.activation(self.time_mlp(time_step_emb))[:, :, np.newaxis, np.newaxis]
            x = x + time_emb

        for layer in self.non_transform_layers:
            x = layer(x)
        return x


class AttentiveSkipConnection(torch.nn.Module):
    def __init__(self, in_channels: int, do_norm: bool = params.DO_NORM) -> None:
        super().__init__()
        self.entry_conv_spatial_downsample = torch.nn.Conv2d(in_channels, in_channels, stride=2, kernel_size=1)
        self.entry_conv_channel_downsample = torch.nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding="same")
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, padding="same")
        self.sigmoid = torch.nn.Sigmoid()
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.exit_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, padding="same")
        if do_norm:
            self.norm = torch.nn.LazyBatchNorm2d()
        else:
            self.norm = torch.nn.Identity()

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N x C x H x W ]
        :param x_skip: [N x C * 2 x H // 2 x W // 2]
        :return: [N x C x H x W]
        """
        spatial_downsample = self.entry_conv_spatial_downsample(x)
        channel_downsample = self.entry_conv_channel_downsample(x_skip)
        x_skip = spatial_downsample + channel_downsample
        x_skip = self.relu(x_skip)
        x_skip = self.conv(x_skip)
        x_skip = self.sigmoid(x_skip)
        x_skip = self.upsample(x_skip)
        x = x * x_skip
        return self.norm(self.exit_conv(x))

