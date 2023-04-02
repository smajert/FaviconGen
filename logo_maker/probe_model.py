from pathlib import Path

from gooey import Gooey, GooeyParser
from matplotlib import pyplot as plt
import torch

from logo_maker.data_loading import show_image_grid, ClusterNamesAeGrayscale
from logo_maker.denoising_diffusion import Generator, draw_sample_from_generator, VarianceSchedule


def probe_model(model_file: Path | ClusterNamesAeGrayscale, seed: int,  n_samples: int, device: str) -> None:
    """
    Sample images from a chosen model.

    :param model_file: Model file (*.pt) or name of a prepared cluster corresponding to a pretrained model
        in 'LogoMaker/data/model' to draw a sample from.
    :param seed: Random seed to start the generation process.
    :param n_samples: Amount of images to draw from the model.
    :param device: Device to use to run the model. Either 'cuda' or 'cpu'.
    """
    if isinstance(model_file, ClusterNamesAeGrayscale):
        model_file = Path(__file__).parents[1] / f"data/models/{model_file.name}/model.pt"

    place_holder_variance_schedule = VarianceSchedule()
    model = Generator(place_holder_variance_schedule)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    batch = draw_sample_from_generator(model, (n_samples, 3, 32, 32), seed=seed)
    show_image_grid(batch)
    plt.show()


@Gooey(
    program_name='LogoMaker',
    image_dir=Path(__file__).parents[1] / "data/gooey_image_dir"
)
def main():
    parser = GooeyParser(description="Get sample images from model")
    parser.add_argument(
        "model",
        choices=[cluster.name for cluster in ClusterNamesAeGrayscale],
        default=ClusterNamesAeGrayscale.round_on_white.name,
        help="Model to use."
    )
    parser.add_argument(
        "seed", type=int, default=42, help="Random number seed to generate Gaussian noise (first timestep) from."
    )
    parser.add_argument("n_samples", type=int, default=1, help="Number of samples to get from model.")

    parser.add_argument(
        "--custom_model_file",
        type=Path,
        default=None,
        widget="FileChooser",
        help=(
            "If you want to use a custom model file (*.pt) instead of the one chosen under the 'model' setting,"
            " select it here."
        )
    )

    parser.add_argument(
        "--use_gpu",
        help="Try to calculate on GPU",
        action="store_true"
    )

    args = parser.parse_args()
    model = args.custom_model_file
    if model is None:
        model = ClusterNamesAeGrayscale[args.model]

    if args.use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    probe_model(model, args.seed, args.n_samples, device)


if __name__ == "__main__":
    main()
