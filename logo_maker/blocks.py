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

        self.time_embedding_dimension = time_embedding_dimension
        if time_embedding_dimension is not None:
            self.time_mlp = torch.nn.Linear(self.time_embedding_dimension, channels_in)

        self.non_transform_layers = torch.nn.ModuleList()
        for _ in range(n_non_transform_conv_layers):
            self.non_transform_layers.extend([
                torch.nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1),
                norm_fn(),
                self.activation
            ])

        if do_transpose:
            self.conv_transform = torch.nn.ConvTranspose2d(
                channels_in, channels_out, kernel_size=kernel, stride=stride, padding=padding
            )
        else:
            self.conv_transform = torch.nn.Conv2d(
                channels_in, channels_out, kernel_size=kernel, stride=stride, padding=padding
            )

    def forward(self, x: torch.Tensor, time_step_emb: torch.Tensor | None = None) -> torch.Tensor:
        if time_step_emb is not None:
            if self.time_embedding_dimension is None:
                raise ValueError("Time step given, but no embedding dimension specified")
            time_emb = self.activation(self.time_mlp(time_step_emb))[:, :, np.newaxis, np.newaxis]
        else:
            time_emb = 0

        for layer_idx, layer in enumerate(self.non_transform_layers):
            if layer_idx == 3:
                x = layer(x + time_emb)
            else:
                x = layer(x)
        return self.activation(self.conv_transform(x))
