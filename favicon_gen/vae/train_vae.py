from pathlib import Path
import shutil

import torch
from tqdm import tqdm

from favicon_gen import params
from favicon_gen.data_loading import load_data, show_image_grid
from favicon_gen.vae.autoencoder import PatchDiscriminator, VariationalAutoEncoder


def train(
    dataset_params: params.Dataset,
    auto_params: params.AutoEncoder,
    general_params: params.General,
    model_file: Path | None = None,
) -> None:
    """
    Training loop for VAE or VAE + adversarial patch discriminator.

    :param dataset_params: Dataset parameters
    :param auto_params: Model parameters
    :param model_file: If given, will start from the model saved there
    :return: Loss for each epoch
    """
    n_samples, data_loader = load_data(general_params.batch_size, dataset_params)
    model_storage_directory = params.OUTS_BASE_DIR

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True, parents=True)

    # prepare discriminator
    use_patch_discriminator = auto_params.adversarial_loss_weight is not None
    if use_patch_discriminator:
        patch_disc = PatchDiscriminator(dataset_params.in_channels)
        patch_disc.to(general_params.device)
        lower_disc_learning_rate = (
            0.1 * general_params.learning_rate
        )  # lower rate helps in GAN training
        optimizer_discriminator = torch.optim.Adam(
            patch_disc.parameters(), lr=lower_disc_learning_rate
        )

    # prepare autoencoder
    n_labels = dataset_params.n_classes
    autoencoder = VariationalAutoEncoder(
        dataset_params.in_channels, n_labels, general_params.embedding_dim
    )
    if model_file is not None:
        autoencoder.load_state_dict(torch.load(model_file))
    autoencoder.to(general_params.device)
    optimizer_generator = torch.optim.Adam(
        autoencoder.parameters(), lr=general_params.learning_rate
    )
    loss_fn = torch.nn.MSELoss()

    running_losses = []
    running_loss = 0
    value_for_original = torch.tensor([1], device=general_params.device, dtype=torch.float)
    value_for_reconstructed = torch.tensor([0], device=general_params.device, dtype=torch.float)
    for epoch in (pbar := tqdm(range(general_params.epochs), desc="Current avg. loss: /, Epochs")):
        batch: torch.Tensor
        for batch, labels in data_loader:
            labels = labels.to(general_params.device)
            batch = batch.to(
                general_params.device
            )  # batch does not track gradients -> does not need to be detached ever

            optimizer_generator.zero_grad()
            reconst_batch, mu, log_var = autoencoder(batch, labels)
            kl_loss = -0.5 * torch.mean(
                1 + log_var - mu.pow(2) - log_var.exp()
            )  # Kullback Leibler loss
            reconstruction_loss = loss_fn(reconst_batch, batch)
            if use_patch_discriminator:
                disc_pred_reconstructed = patch_disc(reconst_batch)
                is_original = torch.broadcast_to(value_for_original, disc_pred_reconstructed.shape)
                is_reconstructed = torch.broadcast_to(
                    value_for_reconstructed, disc_pred_reconstructed.shape
                )
                adversarial_loss = torch.nn.L1Loss()(disc_pred_reconstructed, is_original)
                adversarial_loss_weight = auto_params.adversarial_loss_weight
            else:
                adversarial_loss = 0
                adversarial_loss_weight = 0
            generator_loss = (
                reconstruction_loss
                + auto_params.kl_loss_weight * kl_loss
                + adversarial_loss_weight * adversarial_loss
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
        if (epoch + 1) in [
            int(rel_plot_step * general_params.epochs)
            for rel_plot_step in [0.1, 0.25, 0.5, 0.75, 1.0]
        ]:
            show_image_grid(
                reconst_batch, save_as=model_storage_directory / f"reconstruction_epoch_{epoch}.png"
            )
            show_image_grid(batch, save_as=model_storage_directory / f"original_{epoch}.png")

        pbar.set_description(f"Current avg. loss: {running_loss:.3f}, Epochs")
        running_losses.append(running_loss)
        running_loss = 0

    print(f"Saving model in directory {model_storage_directory} ...")
    torch.save(autoencoder.state_dict(), model_storage_directory / "model.pt")

    return running_losses
