import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from logo_maker.blocks import ConvBlock
from logo_maker.data_loading import ClusterNamesAeGrayscale, load_logos, load_mnist, show_image_grid
import logo_maker.params as params
from logo_maker.utils import q_key_pressed_non_blocking


class AutoEncoder(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.latent_dim = 512
        self.activation = torch.nn.LeakyReLU()
        self.encoder = torch.nn.Sequential(  # input: in_channels x 32 x 32
            ConvBlock(in_channels, 32, self.activation),  # 16 x 16 x 16
            ConvBlock(32, 64, self.activation),  # 16 x 8 x 8
            ConvBlock(64, 128, self.activation),  # 64 x 4 x 4
            torch.nn.Flatten()  # 64*4*4
        )
        flattened_dimension = 128 * 4 * 4
        self.to_latent = torch.nn.Linear(flattened_dimension, self.latent_dim)
        self.to_mu = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.to_log_var = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.from_latent = torch.nn.Linear(self.latent_dim, flattened_dimension)

        self.decoder = torch.nn.Sequential(  # input: 64*4*4
            torch.nn.Unflatten(1, (128, 4, 4)),  # 64 x 4 x 4
            ConvBlock(128, 64, self.activation, do_transpose=True),  # 32 x 8 x 8
            ConvBlock(64, 32, self.activation, do_transpose=True),  # 16 x 16 x 16
            ConvBlock(32, in_channels, self.activation, do_transpose=True),  # in_channels x 32 x 32
            torch.nn.Conv2d(in_channels, in_channels, 5, padding=2, stride=1),  # in_channels x 32 x 32
            torch.nn.Tanh()  # in_channels x 32 x 32
        )

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        encoded_shape = x.shape
        z, mu, log_var = self.convert_to_latent(x)
        x = self.convert_from_latent(z)
        x = torch.reshape(x, shape=encoded_shape)
        x = self.decoder(x)

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
    batch_size: int,
    cluster: ClusterNamesAeGrayscale | None,
    device: str,
    learning_rate: float,
    model_file: Path | None,
    n_epochs: int,
    n_images: int,
    shuffle_data: bool,
    use_mnist: bool
) -> None:
    if use_mnist:
        n_samples, data_loader = load_mnist(batch_size, shuffle_data, n_images)
        model_storage_directory = params.OUTS_BASE_DIR / "train_autoencoder_mnist"
        in_channels = 1
    else:
        n_samples, data_loader = load_logos(batch_size, shuffle_data, n_images, cluster=cluster)
        model_storage_directory = params.OUTS_BASE_DIR / "train_autoencoder_lld"
        in_channels = 3

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True)

    # prepare discriminator
    use_patch_discriminator = params.AutoEncoderParams.ADVERSARIAL_LOSS_WEIGHT is not None
    if use_patch_discriminator:
        patch_disc = PatchDiscriminator(in_channels)
        patch_disc.to(device)
        optimizer_discriminator = torch.optim.Adam(patch_disc.parameters(), lr=0.2 * learning_rate)  # 0.08

    # prepare autoencoder
    autoencoder = AutoEncoder(in_channels)
    if model_file is not None:
        autoencoder.load_state_dict(torch.load(model_file))
    autoencoder.to(device)
    optimizer_generator = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    running_losses = []
    running_loss = 0
    value_for_original = torch.tensor([1], device=device, dtype=torch.float)
    value_for_reconstructed = torch.tensor([0], device=device, dtype=torch.float)
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        for batch_idx, batch in enumerate(data_loader):
            if use_mnist:  # throw away labels from MNIST
                batch = batch[0]
            batch = batch.to(device)  # batch does not track gradients -> does not need to be detached ever

            optimizer_generator.zero_grad()
            reconst_batch, mu, log_var = autoencoder(batch)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())  # Kullblack Leibler loss
            reconstruction_loss = loss_fn(reconst_batch, batch)
            if use_patch_discriminator:
                disc_pred_reconstructed = patch_disc(reconst_batch)
                is_original = torch.broadcast_to(value_for_original, disc_pred_reconstructed.shape)
                is_reconstructed = torch.broadcast_to(value_for_reconstructed, disc_pred_reconstructed.shape)
                adversarial_loss = torch.nn.L1Loss()(disc_pred_reconstructed, is_original)
                adversarial_loss_weight = params.AutoEncoderParams.ADVERSARIAL_LOSS_WEIGHT
            else:
                adversarial_loss = 0
                adversarial_loss_weight = 0
            generator_loss = (
                reconstruction_loss
                + params.AutoEncoderParams.KL_LOSS_WEIGHT * kl_loss
                + adversarial_loss_weight * adversarial_loss
            )
            generator_loss.backward()
            optimizer_generator.step()
            #print(kullblack_leibler_divergence.item()/reconstruction_loss.item())

            if use_patch_discriminator:
                optimizer_discriminator.zero_grad()
                disc_pred_original = patch_disc(batch)
                reconst_batch, _, _ = autoencoder(batch)
                disc_pred_reconstructed = patch_disc(reconst_batch.detach())
                disc_loss = (
                    torch.nn.L1Loss()(disc_pred_original, is_original)
                    + torch.nn.L1Loss()(disc_pred_reconstructed, is_reconstructed)
                )
                disc_loss.backward()
                optimizer_discriminator.step()

            running_loss += generator_loss.item() * batch.shape[0] / n_samples
        if (epoch + 1) in [int(rel_plot_step * n_epochs) for rel_plot_step in [0.1, 0.25, 0.5, 0.75, 1.0]]:
            print(autoencoder.to_log_var.weight.mean())
            print(autoencoder.to_log_var.weight.shape)
            print(f"reconstruction_loss: {reconstruction_loss.item()}")
            print(f"KL divergence: {kl_loss.item()}")
            if params.AutoEncoderParams.ADVERSARIAL_LOSS_WEIGHT is not None:
                print(f"Adversarial loss: {adversarial_loss.item()}")
                print(f"Patch disc loss: {disc_loss.item()}")

            show_image_grid(reconst_batch, save_as=model_storage_directory / f"reconstruction_epoch_{epoch}.png")
            show_image_grid(batch, save_as=model_storage_directory / f"original_{epoch}.png")

        pbar.set_description(f"Current avg. loss: {running_loss:.3f}, Epochs")
        running_losses.append(running_loss)
        running_loss = 0
        if q_key_pressed_non_blocking():
            print("stopping training early and finishing up ...")
            break

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

    model_file = None
    train(
        batch_size=params.AutoEncoderParams.BATCH_SIZE,
        cluster=params.DatasetParams.CLUSTER,
        device=params.DEVICE,
        learning_rate=params.AutoEncoderParams.LEARNING_RATE,
        model_file=model_file,
        n_epochs=params.AutoEncoderParams.EPOCHS,
        n_images=params.DatasetParams.N_IMAGES,
        shuffle_data=params.DatasetParams.SHUFFLE_DATA,
        use_mnist=args.use_mnist
    )
