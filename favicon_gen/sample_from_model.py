"""
Draw samples from both the diffusion model and the VAE
"""

import argparse
import copy
from pathlib import Path
import typing

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from favicon_gen.vae.autoencoder import VariationalAutoEncoder
from favicon_gen.data_loading import show_image_grid, load_data
from favicon_gen.diffusion.custom_model import (
    DiffusionModel,
    diffusion_backward_process,
    VarianceSchedule,
)
from favicon_gen.diffusion.diffuser_model import DiffusersModel
from favicon_gen import params


@torch.no_grad()
def sample_from_vae(
    model_file: Path,
    n_labels: int,
    in_channels: int,
    n_samples: int,
    device: str,
    embedding_dim: int,
    save_as: Path | None = None,
) -> typing.Generator[torch.Tensor, None, None]:
    """
    Draw samples from the Variational AutoEncoder (VAE).

    :param model_file: File where the VAE is saved
    :param n_labels: Amount of different labels in the data (e.g. 10 for the
        10 different digits in MNIST)
    :param in_channels: Amount of channels in input (1 for grayscale MNIST, 3 for color LLD)
    :param n_samples: Amount of images to generate from the model
    :param device: 'cpu' for CPU or 'cuda' for GPU
    :param save_as: If given, plot will be saved here
    :return: Batch of generated images; call again to get more images
    """
    autoencoder = VariationalAutoEncoder(in_channels, n_labels, embedding_dim)
    autoencoder.load_state_dict(torch.load(model_file))
    autoencoder.eval()
    autoencoder.to(device)

    rand_generator = torch.Generator(device=device)

    while True:
        random_latent = torch.randn(
            (n_samples, autoencoder.latent_dim), device=device, generator=rand_generator
        )
        random_labels = autoencoder.label_embedding(
            torch.randint(0, n_labels, size=(n_samples,), device=device, generator=rand_generator)
        )
        batch = autoencoder.decoder(autoencoder.convert_from_latent(random_latent), random_labels)
        if save_as is not None:
            show_image_grid(batch)
            plt.savefig(save_as)

        yield batch


@torch.no_grad()
def sample_from_diffusion_model(
    model_file: Path,
    n_labels: int,
    in_channels: int,
    n_samples: int,
    device: str,
    diffusion_info: params.Diffusion,
    embedding_dim: int,
    save_as: Path | None = None,
) -> typing.Generator[torch.Tensor, None, None]:
    """
    Sample images from the denoising diffusion model.

    :param model_file: File where the diffusion model is saved
    :param n_labels: Amount of different labels in the data (e.g. 10 for the 10 different
        digits in MNIST)
    :param in_channels: Amount of channels in input (1 for grayscale MNIST, 3 for color LLD)
    :param n_samples: Amount of images to generate from the model
    :param device: 'cpu' for CPU or 'cuda' for GPU
    :param diffusion_info: Parameters of the diffusion model
    :param save_as: If given, plot will be saved here
    """

    variance_schedule = VarianceSchedule(
        (diffusion_info.var_schedule_start, diffusion_info.var_schedule_end), diffusion_info.steps
    )
    match diffusion_info.architecture:
        case params.DiffusionArchitecture.CUSTOM:
            generator = DiffusionModel(in_channels, variance_schedule, n_labels, embedding_dim)
        case params.DiffusionArchitecture.UNET2D:
            generator = DiffusersModel(in_channels, variance_schedule, n_labels, 2)
    generator.load_state_dict(torch.load(model_file))
    generator = generator.to(device)
    generator.eval()

    # draw single batch first to set seed
    batch = diffusion_backward_process(
        generator, (n_samples, in_channels, 32, 32), diffusion_info.guiding_factor
    )
    while True:
        if save_as is not None:
            show_image_grid(batch)
            plt.savefig(save_as)
            # plt.show()
        yield batch
        # draw batch without setting seed again
        batch = diffusion_backward_process(
            generator, (n_samples, in_channels, 32, 32), diffusion_info.guiding_factor
        )


@torch.no_grad()
def nearest_neighbor_search(
    generated_batch: torch.Tensor,
    dataset_info: params.Dataset,
    save_as: Path | None = None,
) -> torch.Tensor:
    """
    Search datasets for nearest neighbors of a batch of images (via summed absolute distance).

    :param generated_batch: Images for which to find the nearest neighbors
    :param dataset_info: General information about the dataset
    :param save_as: Save image of the batch of nearest neighbors here
    :return: batch of nearest neighbors
    """
    modified_dataset_info = copy.deepcopy(dataset_info)
    modified_dataset_info.shuffle = False
    _, data_loader = load_data(1, modified_dataset_info)

    nearest_neighbors = torch.zeros(generated_batch.shape, device=generated_batch.device)
    current_nearest_neighbor_distances = torch.full(
        (generated_batch.shape[0],), fill_value=torch.inf, device=generated_batch.device
    )
    # compare every single image from dataset to generated ones and determine how close they are
    for single_image, _ in tqdm(data_loader, desc="Searching dataset for nearest neighbors..."):
        single_image = single_image.to(generated_batch.device)
        # single_image is broadcast along batch dimension
        distances = torch.sum(torch.abs(single_image - generated_batch), dim=(1, 2, 3))
        diffs = distances - current_nearest_neighbor_distances
        closer_neighbor_idxs = (
            diffs < 0
        )  # idx where the current image is a closer neighbor than the current one
        current_nearest_neighbor_distances[closer_neighbor_idxs] = distances[closer_neighbor_idxs]
        nearest_neighbors[closer_neighbor_idxs, ...] = single_image[0, ...]

    if save_as is not None:
        show_image_grid(nearest_neighbors)
        plt.savefig(save_as)

    return nearest_neighbors


def main():
    parser = argparse.ArgumentParser(description="Get sample images from models")
    parser.add_argument(
        "--n_samples", type=int, default=64, help="Number of samples to get from model."
    )
    parser.add_argument("--use_gpu", help="Try to calculate on GPU.", action="store_true")

    args = parser.parse_args()
    if args.use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    config = params.load_config()
    match config.dataset.name:
        case params.AvailableDatasets.MNIST:
            in_channels = 1
            n_labels = 10
        case params.AvailableDatasets.LLD:
            in_channels = 3
            spec_clusters = config.dataset.specific_clusters
            n_labels = 100 if spec_clusters is None else len(spec_clusters)

    model_file = params.OUTS_BASE_DIR / "model.pt"
    samples_out_file = params.OUTS_BASE_DIR / "samples.pdf"

    match config.model:
        case params.AutoEncoder():
            sample_batch = next(
                sample_from_vae(
                    model_file,
                    n_labels,
                    in_channels,
                    args.n_samples,
                    device,
                    config.general.embedding_dim,
                    save_as=samples_out_file,
                )
            )
        case params.Diffusion():
            sample_batch = next(
                sample_from_diffusion_model(
                    model_file,
                    n_labels,
                    in_channels,
                    args.n_samples,
                    device,
                    config.model,
                    config.general.embedding_dim,
                    save_as=samples_out_file,
                )
            )

    nearest_neighbor_search(
        sample_batch,
        config.dataset,
        save_as=params.OUTS_BASE_DIR / "nearest_neighbors.pdf",
    )


if __name__ == "__main__":
    main()
