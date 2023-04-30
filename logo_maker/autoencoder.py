from pathlib import Path
import shutil

import torch
from tqdm import tqdm

from logo_maker.blocks import ConvBlock
from logo_maker.data_loading import ClusterNamesAeGrayscale, LargeLogoDataset, show_image_grid
import logo_maker.params as params


class AutoEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.encoder = torch.nn.Sequential(  # input: 3 x 32 x 32
            ConvBlock(3, 8, self.activation),  # 8 x 16 x 16
            ConvBlock(8, 16, self.activation),  # 16 x 8 x 8
            ConvBlock(16, 3, self.activation, kernel=3, padding=1, stride=1)  # 3 x 8 x 8
        )
        self.decoder = torch.nn.Sequential(  # input: 8 x 8 x 8
            ConvBlock(3, 16, self.activation, do_transpose=True),  # 16 x 16 x 16
            ConvBlock(16, 8, self.activation, do_transpose=True),  # 8 x 32 x 32
            ConvBlock(8, 3, self.activation, kernel=3, padding=1, stride=1),  # 3 x 32 x 32
            torch.nn.Conv2d(3, 3, 1),  # 3 x 32 x 32
            torch.nn.Tanh()  # 3 x 32 x 32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class PatchDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.layers = torch.nn.ModuleList([  # input: 3 x 32 x 32
            ConvBlock(3, 16, self.activation, n_non_transform_conv_layers=0),  # 16 x 16 x 16
            ConvBlock(16, 32, self.activation, n_non_transform_conv_layers=1, kernel=7, padding=3),  # 32 x 16 x 16
            #ConvBlock(32, 8, self.activation, kernel=3, stride=1, padding=1, n_non_transform_conv_layers=1), # 8 x 16 x 16
            torch.nn.Conv2d(32, 1, kernel_size=5, padding=2),  # 1 x 16 x 16
            torch.nn.Sigmoid()
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def train(
    cluster: ClusterNamesAeGrayscale | None,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    model_file: Path | None,
    device: str
) -> None:
    dataset_location = Path(__file__).parents[1] / "data/LLD-icon.hdf5"
    dataset = LargeLogoDataset(dataset_location, cluster=cluster, cache_files=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_storage_directory = params.OUTS_BASE_DIR / "train_autoencoder"

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True)

    # prepare discriminator
    use_patch_discriminator = params.AutoEncoderParams.ADVERSARIAL_LOSS_WEIGHT is not None
    if use_patch_discriminator:
        patch_disc = PatchDiscriminator()
        patch_disc.to(device)
        optimizer_discriminator = torch.optim.Adam(patch_disc.parameters(), lr=learning_rate)

    # prepare autoencoder
    autoencoder = AutoEncoder()
    if model_file is not None:
        autoencoder.load_state_dict(torch.load(model_file))
    autoencoder.to(device)
    optimizer_generator = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    average_losses = []
    running_loss = 0
    value_for_original = torch.tensor([1], device=device, dtype=torch.float)
    value_for_reconstructed = torch.tensor([0], device=device, dtype=torch.float)
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)  # batch does not track gradients -> does not need to be detached ever

            optimizer_generator.zero_grad()
            reconst_batch = autoencoder(batch)
            reconstruction_loss = loss_fn(reconst_batch, batch)
            if use_patch_discriminator:
                disc_pred_reconstructed = patch_disc(reconst_batch)
                is_original = torch.broadcast_to(value_for_original, disc_pred_reconstructed.shape)
                is_reconstructed = torch.broadcast_to(value_for_reconstructed, disc_pred_reconstructed.shape)
                adversarial_loss = loss_fn(disc_pred_reconstructed, is_original)
                adversarial_loss_weight = params.AutoEncoderParams.ADVERSARIAL_LOSS_WEIGHT
            else:
                adversarial_loss = 0
                adversarial_loss_weight = 0
            generator_loss = reconstruction_loss + adversarial_loss * adversarial_loss_weight
            generator_loss.backward()
            optimizer_generator.step()

            if use_patch_discriminator:
                optimizer_discriminator.zero_grad()
                disc_pred_original = patch_disc(batch)
                reconst_batch = autoencoder(batch)
                disc_pred_reconstructed = patch_disc(reconst_batch.detach())
                disc_loss = (
                    loss_fn(disc_pred_original, is_original) + loss_fn(disc_pred_reconstructed, is_reconstructed)
                )
                disc_loss.backward()
                optimizer_discriminator.step()

            running_loss += generator_loss.item()
        if (epoch + 1) % 50 == 0:
            show_image_grid(reconst_batch, save_as=model_storage_directory / f"reconstruction_epoch_{epoch}.png")
            show_image_grid(batch, save_as=model_storage_directory / f"original_{epoch}.png")

        average_loss = running_loss / len(dataset)
        pbar.set_description(f"Current avg. loss: {average_loss:.3f}, Epochs")
        average_losses.append(average_loss)
        running_loss = 0

    print(f"Saving model in directory {model_storage_directory} ...")
    torch.save(autoencoder.state_dict(), model_storage_directory / "model.pt")

    with open(model_storage_directory / "loss.csv", "w") as file:
        file.write("Epoch,Loss\n")
        for epoch, loss in enumerate(average_losses):
            file.write(f"{epoch},{loss}\n")


if __name__ == "__main__":
    model_file = None
    train(
        cluster=params.CLUSTER,
        n_epochs=params.AutoEncoderParams.EPOCHS,
        batch_size=params.AutoEncoderParams.BATCH_SIZE,
        learning_rate=params.AutoEncoderParams.LEARNING_RATE,
        model_file=params.AutoEncoderParams.MODEL_FILE,
        device=params.DEVICE
    )
