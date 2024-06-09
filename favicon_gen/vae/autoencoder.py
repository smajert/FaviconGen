"""
Variational Autoencoder (VAE) with optional patch GAN
"""

from typing import Any  # noqa: F401

import torch

from favicon_gen.blocks import ConvBlock, ResampleModi


class Encoder(torch.nn.Module):
    """
    Encoder part of the VAE. Brings images into the latent dimension.

    :param in_channels: Amount of input channels (1 for grayscale MNIST, 3 for color LLD)
    :param activation: Activation function to use e.g. torch.nn.ReLU()
    """

    def __init__(self, in_channels: int, activation: torch.nn.modules.module.Module) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.activation = activation
        small_kernel = {"kernel_size": 2, "padding": 0}  # type: Any
        # fmt: off
        self.convs = torch.nn.ModuleList([                                        # input: in_channels x 32 x 32
            ConvBlock(in_channels, 16, resample_modus=ResampleModi.NO),           # 16 x 32 x 32
            ConvBlock(16, 32, resample_modus=ResampleModi.DOWN),                  # 32 x 16 x 16
            ConvBlock(32, 64, resample_modus=ResampleModi.DOWN),                  # 64 x 8 x 8
            ConvBlock(64, 128, resample_modus=ResampleModi.DOWN, **small_kernel), # 128 x 4 x 4
            ConvBlock(128, 256, resample_modus=ResampleModi.DOWN, **small_kernel),# 256 x 2 x 2
        ])
        self.flatten = torch.nn.Flatten()                                         # 256*2*2 = 1024
        # fmt: on

    def forward(self, x: torch.Tensor, label_embedding: torch.Tensor) -> torch.Tensor:
        for layer in self.convs:
            x = layer(x, label_embedding)
        x = self.flatten(x)
        return x


class Decoder(torch.nn.Module):
    """
    Decoder part of the VAE. Generates images from a latent

    :param out_channels: Amount of channels in output (1 for grayscale MNIST, 3 for color LLD)
    :param encoder_out_shape: Original shape of encoder output
    :param activation: Activation function to use e.g. torch.nn.ReLU()
    """

    def __init__(
        self,
        out_channels: int,
        encoder_out_shape: tuple[int, ...],
        activation: torch.nn.modules.module.Module,
    ) -> None:
        super().__init__()
        self.activation = activation
        # fmt: off
        self.unflatten = torch.nn.Unflatten(1, encoder_out_shape)                    # 256 x 2 x 2
        small_kernel = {"kernel_size": 2, "padding": 0}  # type: Any
        self.convs = torch.nn.ModuleList([
            ConvBlock(256, 128, resample_modus=ResampleModi.UP, **small_kernel),     # 128 x 4 x 4
            ConvBlock(128, 64, resample_modus=ResampleModi.UP, **small_kernel),      # 64 x 8 x 8
            ConvBlock(64, 32, resample_modus=ResampleModi.UP),                       # 32 x 16 x 16
            ConvBlock(32, 16, resample_modus=ResampleModi.UP),                       # 16 x 32 x 32
            ConvBlock(16, out_channels, resample_modus=ResampleModi.NO),             # out_channels x 32 x 32
        ])
        # fmt: on
        self.last_conv = torch.nn.Conv2d(out_channels, out_channels, 5, padding=2, stride=1)
        self.last_activation = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(x)
        for layer in self.convs:
            x = layer(x, label_embeddings)
        x = self.last_conv(x)
        return self.last_activation(x)


class VariationalAutoEncoder(torch.nn.Module):
    """
    Variational Autoencoder (VAE) for MNIST or LLD.

    :param in_channels: Amount of channels in input (1 for grayscale MNIST, 3 for color LLD)
    :param n_labels: Amount of different labels in the data (e.g. 10 for the 10 different digits in MNIST)
    """

    def __init__(self, in_channels: int, n_labels: int, embedding_dim: int) -> None:
        super().__init__()
        self.latent_dim = 512
        self.activation = torch.nn.LeakyReLU()
        self.label_embedding = torch.nn.Embedding(n_labels, embedding_dim)

        self.encoder = Encoder(in_channels, self.activation)
        encoder_output_shape = (256, 2, 2)
        flattened_dimension = (
            encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2]
        )
        self.to_latent = torch.nn.Linear(flattened_dimension, self.latent_dim)
        self.to_mu = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.to_log_var = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.from_latent = torch.nn.Linear(self.latent_dim, flattened_dimension)

        self.decoder = Decoder(in_channels, encoder_output_shape, self.activation)

    def _reparameterize(self, mu, log_var):
        """
        Apply reparameterization trick to latent.

        :param mu: Latent variables representing mean of the distribution
        :param log_var: Latent variables representing the nat. logarithm of the distribution's variance
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def convert_to_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bring output from encoder into requested latent dimension"""
        z = self.activation(self.to_latent(x))

        mu = self.to_mu(z)
        log_var = self.to_log_var(z)
        z = self._reparameterize(mu, log_var)
        return z, mu, log_var

    def convert_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Enlarge tensor from the latent dimension to the size expected by the decoder"""
        x = self.activation(self.from_latent(z))
        return x

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label_emb = self.label_embedding(labels)
        x = self.encoder(x, label_emb)
        encoded_shape = x.shape
        z, mu, log_var = self.convert_to_latent(x)
        x = self.convert_from_latent(z)
        x = torch.reshape(x, shape=encoded_shape)
        x = self.decoder(x, label_emb)

        return x, mu, log_var


class PatchDiscriminator(torch.nn.Module):
    """
    Small discriminator that can be used as an adversary to the VAE.
    The low amount of parameters seems to improve image output when used
    with the VAE.

    :param in_channels: Amount of channels in input (1 for grayscale MNIST, 3 for color LLD)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # fmt: off
        self.layers = torch.nn.ModuleList([                                # input: 3 x 32 x 32
            torch.nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),# 32 x 16 x 16
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=7, padding=3),                   # 16 x 16 x 16
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 8, kernel_size=5, padding=2),                    # 8 x 16 x 16
            torch.nn.Sigmoid()
        ])
        # fmt: on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
