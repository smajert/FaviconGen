import argparse
from pathlib import Path
import typing

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from favicon_gen.autoencoder import AutoEncoder
from favicon_gen.data_loading import show_image_grid, load_logos, load_mnist, get_number_of_different_labels
from favicon_gen.denoising_diffusion import Generator, draw_sample_from_generator, VarianceSchedule
import favicon_gen.params as params


@torch.no_grad()
def sample_from_autoencoder_model(
    model_file: Path,
    n_labels: int,
    in_channels: int,
    n_samples: int,
    device: str,
    save_as: Path | None = None
) -> typing.Generator[torch.Tensor, None, None]:
    autoencoder = AutoEncoder(in_channels, n_labels)
    autoencoder.load_state_dict(torch.load(model_file))
    autoencoder.eval()
    autoencoder.to(device)

    rand_generator = torch.Generator(device=device)

    while True:
        random_latent = torch.randn((n_samples, autoencoder.latent_dim), device=device, generator=rand_generator)
        random_labels = autoencoder.label_embedding(
            torch.randint(0, n_labels, size=(n_samples, ), device=device, generator=rand_generator)
        )
        batch = autoencoder.decoder(autoencoder.convert_from_latent(random_latent), random_labels)
        if save_as is not None:
            show_image_grid(batch)
            plt.savefig(save_as)
            # plt.show()

        yield batch


@torch.no_grad()
def sample_from_diffusion_model(
    model_file: Path,
    n_labels: int,
    in_channels: int,
    n_samples: int,
    device: str,
    diffusion_info: params.Diffusion = params.Diffusion(),
    save_as: Path | None = None
) -> typing.Generator[torch.Tensor, None, None]:
    """
    Sample images from a chosen model.

    :param seed: Random seed to start the generation process.
    :param n_samples: Amount of images to draw from the model.
    :param device: Device to use to run the model. Either 'cuda' or 'cpu'.
    """

    variance_schedule = VarianceSchedule(
        (diffusion_info.var_schedule_start, diffusion_info.var_schedule_end), diffusion_info.steps
    )
    generator = Generator(in_channels, variance_schedule, n_labels)
    generator.load_state_dict(torch.load(model_file))
    generator = generator.to(device)
    generator.eval()

    # draw single batch first to set seed
    batch = draw_sample_from_generator(generator, (n_samples, in_channels, 32, 32), diffusion_info.guiding_factor)
    while True:
        if save_as is not None:
            show_image_grid(batch)
            plt.savefig(save_as)
            #plt.show()
        yield batch
        # draw batch without setting seed again
        batch = draw_sample_from_generator(generator, (n_samples, in_channels, 32, 32), diffusion_info.guiding_factor)


@torch.no_grad()
def nearest_neighbor_search(
    generated_batch: torch.Tensor,
    n_images: int,
    use_mnist: bool,
    clusters: list[int] | None,
    save_as: Path | None = None
) -> torch.Tensor:
    if use_mnist:
        _, data_loader = load_mnist(1, False, n_images)
    else:
        _, data_loader = load_logos(1, False, n_images, clusters)

    nearest_neighbors = torch.zeros(generated_batch.shape, device=generated_batch.device)
    current_nearest_neighbor_distances = torch.full(
        (generated_batch.shape[0],), fill_value=torch.inf, device=generated_batch.device
    )
    # compare every single image from dataset to generated ones and determine how close they are
    for single_image, _ in tqdm(data_loader, desc="Searching dataset for nearest neighbors..."):
        single_image = single_image.to(generated_batch.device)
        # single_image is broadcast along batch dimension
        distances = torch.sum(torch.abs(single_image - generated_batch), axis=(1, 2, 3))
        diffs = distances - current_nearest_neighbor_distances
        closer_neighbor_idxs = diffs < 0  # idx where the current image is a closer neighbor than the current one
        current_nearest_neighbor_distances[closer_neighbor_idxs] = distances[closer_neighbor_idxs]
        nearest_neighbors[closer_neighbor_idxs, ...] = single_image[0, ...]

    if save_as is not None:
        show_image_grid(nearest_neighbors)
        plt.savefig(save_as)
        #plt.show()

    return nearest_neighbors


def main():
    parser = argparse.ArgumentParser(description="Get sample images from models")
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

    if args.use_mnist:
        model_file_auto = params.OUTS_BASE_DIR / "train_autoencoder_mnist/model.pt"
        save_location_auto_samples = params.OUTS_BASE_DIR / "samples_autoencoder_mnist.pdf"
        model_file_diffusion = params.OUTS_BASE_DIR / "train_diffusion_model_mnist/model.pt"
        save_location_diff_samples = params.OUTS_BASE_DIR / "samples_diffusion_mnist.pdf"
    else:
        model_file_auto = params.OUTS_BASE_DIR / f"train_autoencoder_lld/model.pt"
        save_location_auto_samples = params.OUTS_BASE_DIR / "samples_autoencoder_lld.pdf"
        model_file_diffusion = params.OUTS_BASE_DIR / "train_diffusion_model_lld/model.pt"
        save_location_diff_samples = params.OUTS_BASE_DIR / "samples_diffusion_lld.pdf"

    in_channels = 1 if args.use_mnist else 3
    n_labels = get_number_of_different_labels(args.use_mnist, params.Dataset.clusters)

    auto_gen_batch = next(sample_from_autoencoder_model(
        model_file_auto, n_labels, in_channels, args.n_samples, device, save_as=save_location_auto_samples
    ))
    diffusion_gen_batch = next(sample_from_diffusion_model(
        model_file_diffusion,
        n_labels,
        in_channels,
        args.n_samples,
        device,
        save_as=save_location_diff_samples
    ))

    nearest_neighbor_search(
        auto_gen_batch,
        params.Dataset.n_images,
        args.use_mnist,
        params.Dataset.clusters,
        save_as=params.OUTS_BASE_DIR / f"auto_nearest_neighbors_mnist_{args.use_mnist}.pdf"
    )

    nearest_neighbor_search(
        diffusion_gen_batch,
        params.Dataset.n_images,
        args.use_mnist,
        params.Dataset.clusters,
        save_as=params.OUTS_BASE_DIR / f"diffusion_nearest_neighbors_mnist_{args.use_mnist}.pdf"
    )


if __name__ == "__main__":
    main()
