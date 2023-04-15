import json
from pathlib import Path
import shutil

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import logo_maker.params as params
from logo_maker.data_loading import ClusterNamesAeGrayscale, LargeLogoDataset, show_image_grid


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        activation: torch.nn.modules.activation,
        do_norm: bool = True
    ) -> None:
        super().__init__()
        self.activation = activation

        self.conv_1 = torch.nn.Conv2d(channels, channels, kernel_size, padding="same")
        self.conv_2 = torch.nn.Conv2d(channels, channels, kernel_size, padding="same")

        if do_norm:
            self.norm_1 = torch.nn.LazyBatchNorm2d()
            self.norm_2 = torch.nn.LazyBatchNorm2d()
        else:
            self.norm_1 = torch.nn.Identity()
            self.norm_2 = torch.nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm_1(self.activation(self.conv_1(x)))
        x += residual
        x = self.norm_2(self.activation(self.conv_2(x)))
        return x


class AutoEncoderConvBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        activation: torch.nn.modules.activation,
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
                channels_out, channels_out, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv_3 = torch.nn.Conv2d(
                channels_out, channels_out, kernel_size=4, stride=2, padding=1
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
            ResidualBlock(3, 3, self.activation),
            AutoEncoderConvBlock(3, 8, self.activation),  # 16 x 16 x 16
            ResidualBlock(8, 3, self.activation),
            AutoEncoderConvBlock(8, 16, self.activation),  # 32 x 8 x 8
        ])
        self.decoder = torch.nn.ModuleList([  # input: 3 x 8 x 8
            ResidualBlock(16, 3, self.activation),
            AutoEncoderConvBlock(16, 8, self.activation, do_transpose=True),  # 16 x 16 x 16
            ResidualBlock(8, 3, self.activation),
            AutoEncoderConvBlock(8, 3, self.activation, do_transpose=True),  # 3 x 32 x 32
        ])
        self.last_conv = torch.nn.Conv2d(3, 3, 1)
        self.last_activation = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for encode_layer in self.encoder:
            x = encode_layer(x)

        for decode_layer in self.decoder:
            x = decode_layer(x)

        return self.last_activation(self.last_conv(x))


def train(
    cluster: int | None,
    device="cuda",
    n_epochs: int = 300,
    batch_size: int = 128,
    model_file: Path | None = None
) -> None:
    dataset_location = Path(__file__).parents[1] / "data/LLD-icon.hdf5"
    dataset = LargeLogoDataset(dataset_location, cluster=cluster, cache_files=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_storage_directory = params.OUTS_BASE_DIR / "autoencoder"

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
        model_storage_directory.mkdir()

    model = AutoEncoder()
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    average_losses = []
    running_loss = 0
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}, Batches: ")):
            batch = batch.to(device)
            optimizer.zero_grad()
            # pytorch expects tuple for size here:

            pred = model(batch)
            loss = loss_fn(pred, batch)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            show_image_grid(pred, save_as=model_storage_directory / f"reconstruction_epoch_{epoch}.png")
            show_image_grid(batch, save_as=model_storage_directory / f"original_{epoch}.png")

        average_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{n_epochs}, average loss = {average_loss}")
        average_losses.append(average_loss)
        running_loss = 0

    print(f"Saving model in directory {model_storage_directory} ...")
    torch.save(model.state_dict(), model_storage_directory / "model.pt")

    with open(model_storage_directory / "loss.csv", "w") as file:
        file.write("Epoch,Loss\n")
        for epoch, loss in enumerate(average_losses):
            file.write(f"{epoch},{loss}\n")


if __name__ == "__main__":
    model_file = None
    train(
        ClusterNamesAeGrayscale.round_on_white,
        n_epochs=params.AutoEncoderParams.N_EPOCHS,
        model_file=model_file
    )
