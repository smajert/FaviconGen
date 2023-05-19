import argparse
from datetime import datetime
from pathlib import Path
import random
import typing

from matplotlib import pyplot as plt
import torch

from logo_maker.autoencoder import AutoEncoder
from logo_maker.data_loading import show_image_grid
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
    next(sample_from_autoencoder_model(
        model_file_auto, in_channels, args.seed, args.n_samples, device, save_as=save_location_auto_samples
    ))
    next(sample_from_diffusion_model(
        model_file_diffusion, in_channels, args.seed, args.n_samples, device, save_as=save_location_diff_samples
    ))


if __name__ == "__main__":
    main()
