"""
Variational Autoencoder (VAE) with optional patch GAN
"""

import argparse
from pathlib import Path
import shutil
from typing import Any  # noqa: F401

import torch
from tqdm import tqdm

from favicon_gen.blocks import ConvBlock, ResampleModi
from favicon_gen.data_loading import load_logos, load_mnist, show_image_grid, get_number_of_different_labels
from favicon_gen import params


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
        self.convs = torch.nn.ModuleList([                                          # input: in_channels x 32 x 32
            ConvBlock(in_channels, 16, resample_modus=ResampleModi.NO),             # 16 x 32 x 32
            ConvBlock(16, 32, resample_modus=ResampleModi.DOWN),                    # 32 x 16 x 16
            ConvBlock(32, 64, resample_modus=ResampleModi.DOWN),                    # 64 x 8 x 8
            ConvBlock(64, 128, resample_modus=ResampleModi.DOWN, **small_kernel),   # 128 x 4 x 4
            ConvBlock(128, 256, resample_modus=ResampleModi.DOWN, **small_kernel),  # 256 x 2 x 2
        ])
        self.flatten = torch.nn.Flatten()                                           # 256*2*2 = 1024
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
        self, out_channels: int, encoder_out_shape: tuple[int, ...], activation: torch.nn.modules.module.Module
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

    def __init__(self, in_channels: int, n_labels: int) -> None:
        super().__init__()
        self.latent_dim = 512
        self.activation = torch.nn.LeakyReLU()
        embedding_dim = params.EMBEDDING_DIM
        self.label_embedding = torch.nn.Embedding(n_labels, embedding_dim)

        self.encoder = Encoder(in_channels, self.activation)
        encoder_output_shape = (256, 2, 2)
        flattened_dimension = encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2]
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

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.layers = torch.nn.ModuleList([                                             # input: 3 x 32 x 32
            torch.nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),       # 32 x 16 x 16
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=7, padding=3),                          # 16 x 16 x 16
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 8, kernel_size=5, padding=2),                           # 8 x 16 x 16
            torch.nn.Sigmoid()
        ])
        # fmt: on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def train(
    dataset_info: params.Dataset,
    auto_info: params.AutoEncoder,
    use_mnist: bool,
    model_file: Path | None = None,
) -> None:
    """
    Training loop for VAE or VAE + adversarial patch discriminator.

    :param dataset_info: Dataset parameters
    :param auto_info: Model parameters
    :param use_mnist: Whether to use MNIST (`True`) or LLD (`False`) as training data
    :param model_file: If given, will start from the model saved there
    """
    if use_mnist:
        n_samples, data_loader = load_mnist(auto_info.batch_size, dataset_info.shuffle, dataset_info.n_images)
        n_epochs = auto_info.epochs_mnist
        model_storage_directory = params.OUTS_BASE_DIR / "train_autoencoder_mnist"
        in_channels = 1
    else:
        n_samples, data_loader = load_logos(
            auto_info.batch_size, dataset_info.shuffle, dataset_info.n_images, clusters=dataset_info.specific_clusters
        )
        n_epochs = auto_info.epochs_lld
        model_storage_directory = params.OUTS_BASE_DIR / "train_autoencoder_lld"
        in_channels = 3

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True)

    # prepare discriminator
    use_patch_discriminator = auto_info.adversarial_loss_weight is not None
    if use_patch_discriminator:
        patch_disc = PatchDiscriminator(in_channels)
        patch_disc.to(params.DEVICE)
        lower_disc_learning_rate = 0.1 * auto_info.learning_rate  # lower rate helps in GAN training
        optimizer_discriminator = torch.optim.Adam(patch_disc.parameters(), lr=lower_disc_learning_rate)

    # prepare autoencoder
    n_labels = get_number_of_different_labels(use_mnist, dataset_info.specific_clusters)
    autoencoder = VariationalAutoEncoder(in_channels, n_labels)
    if model_file is not None:
        autoencoder.load_state_dict(torch.load(model_file))
    autoencoder.to(params.DEVICE)
    optimizer_generator = torch.optim.Adam(autoencoder.parameters(), lr=auto_info.learning_rate)
    loss_fn = torch.nn.MSELoss()

    running_losses = []
    running_loss = 0
    value_for_original = torch.tensor([1], device=params.DEVICE, dtype=torch.float)
    value_for_reconstructed = torch.tensor([0], device=params.DEVICE, dtype=torch.float)
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        batch: torch.Tensor
        for batch, labels in data_loader:
            labels = labels.to(params.DEVICE)
            batch = batch.to(params.DEVICE)  # batch does not track gradients -> does not need to be detached ever

            optimizer_generator.zero_grad()
            reconst_batch, mu, log_var = autoencoder(batch, labels)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())  # Kullback Leibler loss
            reconstruction_loss = loss_fn(reconst_batch, batch)
            if use_patch_discriminator:
                disc_pred_reconstructed = patch_disc(reconst_batch)
                is_original = torch.broadcast_to(value_for_original, disc_pred_reconstructed.shape)
                is_reconstructed = torch.broadcast_to(value_for_reconstructed, disc_pred_reconstructed.shape)
                adversarial_loss = torch.nn.L1Loss()(disc_pred_reconstructed, is_original)
                adversarial_loss_weight = auto_info.adversarial_loss_weight
            else:
                adversarial_loss = 0
                adversarial_loss_weight = 0
            generator_loss = (
                reconstruction_loss + auto_info.kl_loss_weight * kl_loss + adversarial_loss_weight * adversarial_loss
            )
            generator_loss.backward()
            optimizer_generator.step()

            if use_patch_discriminator:
                optimizer_discriminator.zero_grad()
                disc_pred_original = patch_disc(batch)
                reconst_batch, _, _ = autoencoder(batch, labels)
                disc_pred_reconstructed = patch_disc(reconst_batch.detach())
                disc_loss = torch.nn.L1Loss()(disc_pred_original, is_original) + torch.nn.L1Loss()(
                    disc_pred_reconstructed, is_reconstructed
                )
                disc_loss.backward()
                optimizer_discriminator.step()

            running_loss += generator_loss.item() * batch.shape[0] / n_samples
        if (epoch + 1) in [int(rel_plot_step * n_epochs) for rel_plot_step in [0.1, 0.25, 0.5, 0.75, 1.0]]:
            show_image_grid(reconst_batch, save_as=model_storage_directory / f"reconstruction_epoch_{epoch}.png")
            show_image_grid(batch, save_as=model_storage_directory / f"original_{epoch}.png")

        pbar.set_description(f"Current avg. loss: {running_loss:.3f}, Epochs")
        running_losses.append(running_loss)
        running_loss = 0

    print(f"Saving model in directory {model_storage_directory} ...")
    torch.save(autoencoder.state_dict(), model_storage_directory / "model.pt")

    with open(model_storage_directory / "loss.csv", "w", encoding="utf-8") as file:
        file.write("Epoch,Loss\n")
        for epoch, loss in enumerate(running_losses):
            file.write(f"{epoch},{loss}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the autoencoder model.")
    parser.add_argument(
        "--use_mnist", action="store_true", help="Whether to train on MNIST instead of the Large Logo Dataset."
    )
    args = parser.parse_args()

    train(dataset_info=params.Dataset(), auto_info=params.AutoEncoder(), use_mnist=args.use_mnist)
