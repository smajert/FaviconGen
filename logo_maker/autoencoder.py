from pathlib import Path
import shutil

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import logo_maker.params as params
from logo_maker.data_loading import LargeLogoDataset, show_image_grid


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        activation: torch.nn.modules.activation,
        kernel: int | tuple[int, int] = 4,
        stride: int = 2,
        padding: int = 1,
        do_norm: bool = True,
        do_transpose: bool = False,
    ) -> None:
        super().__init__()

        if do_norm:
            self.norm_1 = torch.nn.LazyBatchNorm2d()
            self.norm_2 = torch.nn.LazyBatchNorm2d()
        else:
            self.norm_1 = torch.nn.Identity()
            self.norm_2 = torch.nn.Identity()

        self.conv_1 = torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)  # width/height stay same
        self.conv_2 = torch.nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)  # width/height stay same
        if do_transpose:
            self.conv_3 = torch.nn.ConvTranspose2d(
                channels_out, channels_out, kernel_size=kernel, stride=stride, padding=padding
            )
        else:
            self.conv_3 = torch.nn.Conv2d(
                channels_out, channels_out, kernel_size=kernel, stride=stride, padding=padding
            )

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.norm_1(self.conv_1(x)))
        x = self.activation(self.norm_2(self.conv_2(x)))
        return self.activation(self.conv_3(x))


class AutoEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.encoder = torch.nn.ModuleList([  # input: 3 x 32 x 32
            ConvBlock(3, 8, self.activation),  # 8 x 16 x 16
            ConvBlock(8, 16, self.activation),  # 16 x 8 x 8
        ])
        self.decoder = torch.nn.ModuleList([  # input: 16 x 8 x 8
            ConvBlock(16, 8, self.activation, do_transpose=True),  # 8 x 16 x 16
            ConvBlock(8, 3, self.activation, do_transpose=True),  # 3 x 32 x 32
        ])
        self.last_conv = torch.nn.Conv2d(3, 3, 1)
        self.last_activation = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for encode_layer in self.encoder:
            x = encode_layer(x)

        for decode_layer in self.decoder:
            x = decode_layer(x)

        return self.last_activation(self.last_conv(x))


class PatchDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.layers = torch.nn.ModuleList([  # input: 3 x 32 x 32
            ConvBlock(3, 16, self.activation),  # 16 x 16 x 16
            ConvBlock(16, 32, self.activation),  # 32 x 8 x 8
            ConvBlock(32, 8, self.activation, kernel=3, stride=1, padding=1),  # 8 x 8 x 8
            torch.nn.Conv2d(8, 1, kernel_size=3, padding=1), # 1 x 8 x 8
            torch.nn.Softmax()
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def train(
    cluster: int | None,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    model_file: Path | None,
    device="cuda"
) -> None:
    dataset_location = Path(__file__).parents[1] / "data/LLD-icon.hdf5"
    dataset = LargeLogoDataset(dataset_location, cluster=cluster, cache_files=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_storage_directory = params.OUTS_BASE_DIR / "train_autoencoder"

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
        model_storage_directory.mkdir()

    autoencoder = AutoEncoder()
    if model_file is not None:
        autoencoder.load_state_dict(torch.load(model_file))

    patch_disc = PatchDiscriminator()

    autoencoder.to(device)
    patch_disc.to(device)
    optimizer = torch.optim.Adam(list(autoencoder.parameters()) + list(patch_disc.parameters()), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    average_losses = []
    running_loss = 0
    value_for_original = torch.tensor([1], device=device, dtype=torch.float)
    value_for_reconstructed = torch.tensor([0], device=device, dtype=torch.float)
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            # pytorch expects tuple for size here:

            reconst_batch = autoencoder(batch)
            reconstruction_loss = loss_fn(reconst_batch, batch)
            disc_pred_original = patch_disc(batch)
            disc_pred_reconstructed = patch_disc(reconst_batch)
            is_original = torch.broadcast_to(value_for_original, disc_pred_original.shape)
            is_reconstructed = torch.broadcast_to(value_for_reconstructed, disc_pred_reconstructed.shape)
            disc_loss = loss_fn(disc_pred_original, is_original) + loss_fn(disc_pred_reconstructed, is_reconstructed)

            loss = reconstruction_loss + disc_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 5 == 0:
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
        n_epochs=params.AutoEncoderParams.N_EPOCHS,
        batch_size=params.AutoEncoderParams.BATCH_SIZE,
        learning_rate=params.AutoEncoderParams.LEARNING_RATE,
        model_file=params.AutoEncoderParams.MODEL_FILE
    )
