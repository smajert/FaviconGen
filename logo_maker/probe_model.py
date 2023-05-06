import argparse
from pathlib import Path

#from gooey import Gooey, GooeyParser
from matplotlib import pyplot as plt
import torch

from logo_maker.autoencoder import AutoEncoder
from logo_maker.data_loading import show_image_grid
from logo_maker.denoising_diffusion import Generator, draw_sample_from_generator, VarianceSchedule
import logo_maker.params as params


def probe_autoencoder_model(seed: int, n_samples: int, device:str, save_as: Path | None = None) -> None:
    autoencoder_file = params.OUTS_BASE_DIR / f"train_autoencoder/model.pt"
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(autoencoder_file))
    autoencoder.eval()
    autoencoder.to(device)

    random_latent = torch.randn((n_samples, 64, 8, 8), device=device)#, seed=seed)
    batch = autoencoder.decoder(autoencoder.convert_from_latent(random_latent))
    show_image_grid(batch)
    if save_as is not None:
        plt.savefig(save_as)
    else:
        plt.savefig(params.OUTS_BASE_DIR / "samples.png")
    plt.show()


def probe_diffusion_model(seed: int, n_samples: int, device: str, save_as: Path | None = None) -> None:
    """
    Sample images from a chosen model.

    :param seed: Random seed to start the generation process.
    :param n_samples: Amount of images to draw from the model.
    :param device: Device to use to run the model. Either 'cuda' or 'cpu'.
    """

    autoencoder_file = params.OUTS_BASE_DIR / f"train_autoencoder/model.pt"
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(autoencoder_file))
    autoencoder.eval()
    autoencoder.to(device)

    generator_file = params.OUTS_BASE_DIR / f"train_diffusion_model/model.pt"
    variance_schedule = VarianceSchedule(
        (params.DiffusionModelParams.VAR_SCHEDULE_START, params.DiffusionModelParams.VAR_SCHEDULE_END),
        params.DiffusionModelParams.DIFFUSION_STEPS
    )
    generator = Generator(variance_schedule, params.DiffusionModelParams.EMBEDDING_DIMENSION)
    generator.load_state_dict(torch.load(generator_file))
    generator = generator.to(device)
    generator.eval()

    batch = draw_sample_from_generator(generator, autoencoder, (n_samples, 3, 8, 8), seed=seed)
    show_image_grid(batch)
    if save_as is not None:
        plt.savefig(save_as)
    else:
        plt.savefig(params.OUTS_BASE_DIR / "samples.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Get sample images from models")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random number seed to generate Gaussian noise (first timestep) from."
    )
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to get from model.")
    parser.add_argument(
        "--use_gpu",
        help="Try to calculate on GPU.",
        action="store_true"
    )
    parser.add_argument(
        "--save_to",
        help="Storage directory for results.",
        type=Path,
        default=None
    )

    args = parser.parse_args()
    if args.use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if args.save_to is not None:
        if not args.save_to.isdir():
            raise OSError(f"{args.save_to} is not a valid directory.")
        save_location_auto_samples = args.save_to / f"samples_autoencoder.png"
        save_location_diff_samples = args.save_to / f"samples_diffusion.png"
    else:
        save_location_auto_samples, save_location_diff_samples = None, None

    probe_autoencoder_model(args.seed, args.n_samples, device, save_as=save_location_auto_samples)
    #probe_diffusion_model(args.seed, args.n_samples, device, save_as=save_location_diff_samples)


if __name__ == "__main__":
    main()
