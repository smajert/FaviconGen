from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from favicon_gen import params
from favicon_gen.data_loading import load_data
from favicon_gen.diffusion.custom_model import (
    DiffusionModel,
    diffusion_forward_process,
    diffusion_backward_process,
    VarianceSchedule,
)
from favicon_gen.diffusion.diffuser_model import DiffusersModel


def train(
    dataset_info: params.Dataset,
    diffusion_info: params.Diffusion,
    general_params: params.General,
    model_file: Path | None = None,
) -> np.ndarray:
    """
    Training loop for diffusion model.

    :param dataset_info: Dataset parameters
    :param diffusion_info: Model parameters
    :param model_file: If given, will start from the model saved there
    :return: Loss for each epoch
    """

    n_samples, data_loader = load_data(general_params.batch_size, dataset_info)
    model_storage_directory = params.OUTS_BASE_DIR

    print(f"Cleaning output directory {model_storage_directory} ...")
    if model_storage_directory.exists():
        shutil.rmtree(model_storage_directory)
    model_storage_directory.mkdir(exist_ok=True)

    beta_start_end = (diffusion_info.var_schedule_start, diffusion_info.var_schedule_end)
    schedule = VarianceSchedule(
        beta_start_end=beta_start_end,
        n_time_steps=diffusion_info.steps,
        device=general_params.device,
    )

    n_labels = dataset_info.n_classes
    match diffusion_info.architecture:
        case params.DiffusionArchitecture.CUSTOM:
            model = DiffusionModel(
                dataset_info.in_channels, schedule, n_labels, general_params.embedding_dim
            )
        case params.DiffusionArchitecture.UNET2D:
            model = DiffusersModel(dataset_info.in_channels, schedule, n_labels, 2)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.to(general_params.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=general_params.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=general_params.lr_reduction_patience,
        verbose=True,
        min_lr=5e-6,
    )
    loss_fn = torch.nn.MSELoss()
    running_losses = []
    running_loss = 0
    n_epochs = general_params.epochs
    for epoch in (pbar := tqdm(range(n_epochs), desc="Current avg. loss: /, Epochs")):
        for batch, labels in data_loader:
            labels = labels.to(general_params.device)
            if np.random.random() < 0.1:
                labels = None
            batch = batch.to(general_params.device)

            optimizer.zero_grad()
            # pytorch expects tuple for size here:
            actual_batch_size = batch.shape[0]
            t = torch.randint(low=0, high=diffusion_info.steps, size=(actual_batch_size,))
            noisy_batch, noise = diffusion_forward_process(batch, t, schedule)

            noise_pred = model(noisy_batch, t.to(general_params.device), labels)
            loss = loss_fn(noise_pred, noise)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.shape[0] / n_samples

        lr_scheduler.step(loss)
        if (epoch + 1) in [
            int(rel_plot_step * n_epochs) for rel_plot_step in [0.1, 0.25, 0.5, 0.75, 1.0]
        ]:
            sample_shape = torch.Size((1, *batch.shape[1:]))
            _ = diffusion_backward_process(
                model,
                sample_shape,
                diffusion_info.guiding_factor,
                save_sample_as=model_storage_directory / f"epoch_{epoch + 1}.png",
            )

        pbar.set_description(f"Current avg. loss: {running_loss:.3f}, Epochs")
        running_losses.append(running_loss)
        running_loss = 0

    torch.save(model.state_dict(), model_storage_directory / "model.pt")

    return running_losses
