import numpy as np
import torch

import logo_maker.params as params


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


class SelfAttention(torch.nn.Module):
    """ taken from https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py"""
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = torch.nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = torch.nn.LayerNorm([channels])
        self.ff_self = torch.nn.Sequential(
            torch.nn.LayerNorm([channels]),
            torch.nn.Linear(channels, channels),
            torch.nn.GELU(),
            torch.nn.Linear(channels, channels),
        )

    def forward(self, x):
        if len(x.shape) == 4:
            n_height = x.shape[2]
            n_width = x.shape[3]
        else:
            raise ValueError(
                "Input to SelfAttention should be four dimensional (batch x channel x width x height),"
                f"but given shape is {x.shape}."
            )
        x = x.view(-1, self.channels, n_height * n_width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, n_height, n_width)