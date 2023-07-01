import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from logo_maker.blocks import ConvBlock
from logo_maker.data_loading import load_logos, load_mnist, show_image_grid
import logo_maker.params as params


class Encoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, embedding_dim: int, n_labels: int, activation: torch.nn.modules.activation
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.label_embedding = torch.nn.Embedding(n_labels, embedding_dim)
        self.convs = torch.nn.ModuleList([
            ConvBlock(in_channels, 32, self.activation, time_embedding_dimension=embedding_dim),  # 32 x 16 x 16
            ConvBlock(32, 64, self.activation, time_embedding_dimension=embedding_dim),  # 64 x 8 x 8
            ConvBlock(64, 128, self.activation, time_embedding_dimension=embedding_dim),  # 128 x 4 x 4
            ConvBlock(128, 256, self.activation, time_embedding_dimension=embedding_dim),  # 256 x 2 x 2
        ])
        self.flatten = torch.nn.Flatten()  # 256*2*2

    def forward(self, x: torch.Tensor, label_embedding: torch.Tensor) -> torch.Tensor:
        for layer in self.convs:
            x = layer(x, label_embedding)
        x = self.flatten(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(
        self, out_channels: int, embedding_dim, batch_shape: tuple[int, ...], activation: torch.nn.modules.activation
    ) -> None:
        super().__init__()
        self.activation = activation
        self.unflatten = torch.nn.Unflatten(1, batch_shape)  # 256 x 2 x 2
        self.convs = torch.nn.ModuleList([
            ConvBlock(256, 128, self.activation, do_transpose=True, time_embedding_dimension=embedding_dim),  # 128 x 4 x 4
            ConvBlock(128, 64, self.activation, do_transpose=True, time_embedding_dimension=embedding_dim),  # 64 x 8 x 8
            ConvBlock(64, 32, self.activation, do_transpose=True, time_embedding_dimension=embedding_dim),  # 64 x 16 x 16
            ConvBlock(32, out_channels, self.activation, do_transpose=True, time_embedding_dimension=embedding_dim),  # in_channels x 32 x 32
        ])
        self.last_conv = torch.nn.Conv2d(out_channels, out_channels, 5, padding=2, stride=1)  # in_channels x 32 x 32
        self.last_activation = torch.nn.Tanh()  # in_channels x 32 x 32

    def forward(self, x: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(x)
        for layer in self.convs:
            x = layer(x, label_embeddings)
        x = self.last_conv(x)
        return self.last_activation(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, n_labels: int) -> None:
        super().__init__()
        self.latent_dim = 512
        self.activation = torch.nn.LeakyReLU()
        self.label_embedding = torch.nn.Embedding(n_labels, embedding_dim)

        self.encoder = Encoder(in_channels, embedding_dim, n_labels, self.activation)
        flattened_dimension = 256 * 2 * 2
        self.to_latent = torch.nn.Linear(flattened_dimension, self.latent_dim)
        self.to_mu = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.to_log_var = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.from_latent = torch.nn.Linear(self.latent_dim, flattened_dimension)

        self.decoder = Decoder(in_channels, embedding_dim, (256, 2, 2), self.activation)

    def _reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def convert_to_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.activation(self.to_latent(x))

        # get `mu` and `log_var`
        mu = self.to_mu(z)
        log_var = self.to_log_var(z)
        z = self._reparameterize(mu, log_var)
        return z, mu, log_var

    def convert_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.from_latent(z))
        return x

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label_emb = self.label_embedding(labels)
        x = self.encoder(x, label_emb)
        encoded_shape = x.shape
        z, mu, log_var = self.convert_to_latent(x)
        x = self.convert_from_latent(z)
        x = torch.reshape(x, shape=encoded_shape)
        x = self.decoder(x, label_emb)

        return x, mu, log_var


class PatchDiscriminator(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.layers = torch.nn.ModuleList([  # input: 3 x 32 x 32
            ConvBlock(in_channels, 16, self.activation, n_non_transform_conv_layers=0),  # 16 x 16 x 16
            ConvBlock(16, 32, self.activation, n_non_transform_conv_layers=0, kernel=7, padding=3),  # 32 x 8 x 8
            torch.nn.Conv2d(32, 8, kernel_size=5, padding=2),  # 8 x 8 x 8
            torch.nn.Sigmoid()
        ])

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
    if use_mnist:
        n_samples, data_loader = load_mnist(auto_info.batch_size, dataset_info.shuffle, dataset_info.n_images)
        n_epochs = auto_info.epochs_mnist
        model_storage_directory = params.OUTS_BASE_DIR / "train_autoencoder_mnist"
        in_channels = 1
    else:
        n_samples, data_loader = load_logos(
            auto_info.batch_size, dataset_info.shuffle, dataset_info.n_images, cluster=dataset_info.cluster
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
        optimizer_discriminator = torch.optim.Adam(patch_disc.parameters(), lr=0.2 * auto_info.learning_rate)

    # prepare autoencoder
    n_labels = 10 if use_mnist else 100
    autoencoder = AutoEncoder(in_channels, auto_info.embedding_dim, n_labels)
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
        for batch_idx, (batch, labels) in enumerate(data_loader):
            labels = labels.to(params.DEVICE)
            batch = batch.to(params.DEVICE)  # batch does not track gradients -> does not need to be detached ever

            optimizer_generator.zero_grad()
            reconst_batch, mu, log_var = autoencoder(batch, labels)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())  # Kullblack Leibler loss
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
                reconstruction_loss
                + auto_info.kl_loss_weight * kl_loss
                + adversarial_loss_weight * adversarial_loss
            )
            generator_loss.backward()
            optimizer_generator.step()

            if use_patch_discriminator:
                optimizer_discriminator.zero_grad()
                disc_pred_original = patch_disc(batch)
                reconst_batch, _, _ = autoencoder(batch, labels)
                disc_pred_reconstructed = patch_disc(reconst_batch.detach())
                disc_loss = (
                    torch.nn.L1Loss()(disc_pred_original, is_original)
                    + torch.nn.L1Loss()(disc_pred_reconstructed, is_reconstructed)
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

    with open(model_storage_directory / "loss.csv", "w") as file:
        file.write("Epoch,Loss\n")
        for epoch, loss in enumerate(running_losses):
            file.write(f"{epoch},{loss}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the autoencoder model.")
    parser.add_argument(
        "--use_mnist", action="store_true", help="Whether to train on MNIST instead of the Large Logo Dataset."
    )
    args = parser.parse_args()

    train(
        dataset_info=params.Dataset(),
        auto_info=params.AutoEncoder(),
        use_mnist=args.use_mnist
    )
