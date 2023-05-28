import argparse
from datetime import datetime
from pathlib import Path
import random
import typing

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from logo_maker.autoencoder import AutoEncoder
from logo_maker.data_loading import show_image_grid, load_logos, load_mnist
from logo_maker.denoising_diffusion import Generator, draw_sample_from_generator, VarianceSchedule
import logo_maker.params as params


@torch.no_grad()
def sample_from_autoencoder_model(
    model_file: Path, in_channels: int, seed: int | None, n_samples: int, device: str, save_as: Path | None = None
) -> typing.Generator[torch.Tensor, None, None]:
    autoencoder = AutoEncoder(in_channels)
    autoencoder.load_state_dict(torch.load(model_file))
    autoencoder.eval()
    autoencoder.to(device)

    rand_generator = torch.Generator(device=device)
    if seed is not None:
        rand_generator.manual_seed(seed)

    while True:
        random_latent = torch.randn((n_samples, autoencoder.latent_dim), device=device, generator=rand_generator)
        batch = autoencoder.decoder(autoencoder.convert_from_latent(random_latent))
        if save_as is not None:
            show_image_grid(batch)
            plt.savefig(save_as)
            plt.show()

        yield batch


@torch.no_grad()
def sample_from_diffusion_model(
    model_file: Path, in_channels: int,  seed: int, n_samples: int, device: str, save_as: Path | None = None
) -> typing.Generator[torch.Tensor, None, None]:
    """
    Sample images from a chosen model.

    :param seed: Random seed to start the generation process.
    :param n_samples: Amount of images to draw from the model.
    :param device: Device to use to run the model. Either 'cuda' or 'cpu'.
    """

    variance_schedule = VarianceSchedule(
        (params.DiffusionModelParams.VAR_SCHEDULE_START, params.DiffusionModelParams.VAR_SCHEDULE_END),
        params.DiffusionModelParams.DIFFUSION_STEPS
    )
    generator = Generator(in_channels, variance_schedule, params.DiffusionModelParams.EMBEDDING_DIMENSION)
    generator.load_state_dict(torch.load(model_file))
    generator = generator.to(device)
    generator.eval()

    # draw single batch first to set seed
    batch = draw_sample_from_generator(generator, (n_samples, in_channels, 32, 32), seed=seed)
    while True:
        if save_as is not None:
            show_image_grid(batch)
            plt.savefig(save_as)
            plt.show()
        yield batch
        # draw batch without setting seed again
        batch = draw_sample_from_generator(generator, (n_samples, in_channels, 32, 32))


@torch.no_grad()
def nearest_neighbor_search(
    generated_batch: torch.Tensor,
    n_images: int,
    use_mnist: bool,
    cluster: params.ClusterNamesAeGrayscale | None,
    save_as: Path | None = None
) -> torch.Tensor:
    if use_mnist:
        _, data_loader = load_mnist(1, False, n_images)
    else:
        _, data_loader = load_logos(1, False, n_images, cluster)

    nearest_neighbors = torch.zeros(generated_batch.shape, device=generated_batch.device)
    current_nearest_neighbor_distances = torch.full(
        (generated_batch.shape[0],), fill_value=torch.inf, device=generated_batch.device
    )
    # compare every single image from dataset to generated ones and determine how close they are
    for single_image in tqdm(data_loader, desc="Searching dataset for nearest neighbors..."):
        if use_mnist:
            single_image = single_image[0]
        single_image = single_image.to(generated_batch.device)
        # single_image is broadcast along batch dimension
        distances = torch.sum(torch.abs(single_image - generated_batch), axis=(1, 2, 3))
        diffs = distances - current_nearest_neighbor_distances
        closer_neighbor_idxs = diffs < 0 # idx where the current image is a closer neighbor than the current one
        current_nearest_neighbor_distances[closer_neighbor_idxs] = distances[closer_neighbor_idxs]
        nearest_neighbors[closer_neighbor_idxs, ...] = single_image[0, ...]

    if save_as is not None:
        show_image_grid(nearest_neighbors)
        plt.savefig(save_as)
        plt.show()

    return nearest_neighbors


def main():
    parser = argparse.ArgumentParser(description="Get sample images from models")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random number seed to generate Gaussian noise (first timestep) from."
    )
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples to get from model.")
    parser.add_argument(
        "--use_gpu",
        help="Try to calculate on GPU.",
        action="store_true"
    )
    parser.add_argument(
        "--use_mnist", action="store_true", help="Whether to train on MNIST instead of the Large Logo Dataset."
    )

    args = parser.parse_args()
    if args.use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if args.seed is None:
        torch.random.manual_seed(datetime.now().timestamp())
        random.seed(datetime.now().timestamp())

    if args.use_mnist:
        model_file_auto = params.OUTS_BASE_DIR / "train_autoencoder_mnist/model.pt"
        save_location_auto_samples = params.OUTS_BASE_DIR / "samples_autoencoder_mnist.png"
        model_file_diffusion = params.OUTS_BASE_DIR / "train_diffusion_model_mnist/model.pt"
        save_location_diff_samples = params.OUTS_BASE_DIR / "samples_diffusion_mnist.png"
    else:
        model_file_auto = params.OUTS_BASE_DIR / f"train_autoencoder_lld/model.pt"
        save_location_auto_samples = params.OUTS_BASE_DIR / "samples_autoencoder_lld.png"
        model_file_diffusion = params.OUTS_BASE_DIR / "train_diffusion_model_lld/model.pt"
        save_location_diff_samples = params.OUTS_BASE_DIR / "samples_diffusion_lld.png"

    in_channels = 1 if args.use_mnist else 3
    auto_gen_batch = next(sample_from_autoencoder_model(
        model_file_auto, in_channels, args.seed, args.n_samples, device, save_as=save_location_auto_samples
    ))
    diffusion_gen_batch = next(sample_from_diffusion_model(
        model_file_diffusion, in_channels, args.seed, args.n_samples, device, save_as=save_location_diff_samples
    ))

    nearest_neighbor_search(
        auto_gen_batch,
        params.DatasetParams.N_IMAGES,
        args.use_mnist,
        params.DatasetParams.CLUSTER,
        save_as=params.OUTS_BASE_DIR / f"auto_nearest_neighbors_mnist_{args.use_mnist}.png"
    )

    nearest_neighbor_search(
        diffusion_gen_batch,
        params.DatasetParams.N_IMAGES,
        args.use_mnist,
        params.DatasetParams.CLUSTER,
        save_as=params.OUTS_BASE_DIR / f"diffusion_nearest_neighbors_mnist_{args.use_mnist}.png"
    )


if __name__ == "__main__":
    main()