from pathlib import Path

from logo_maker.data_loading import show_image_grid
from logo_maker.stable_diffusion import Generator, draw_sample_from_generator, VarianceSchedule
from matplotlib import pyplot as plt
import torch


def probe_model(model_file: Path, n_samples: int = 32) -> None:
    place_holder_variance_schedule = VarianceSchedule()
    model = Generator(place_holder_variance_schedule)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    batch = draw_sample_from_generator(model, (n_samples, 3, 32, 32))
    show_image_grid(batch)
    plt.show()


if __name__ == "__main__":
    model_file = Path(r"C:\Users\steph\AppData\Local\Temp\logo_nadn_1sh\model.pt")
    probe_model(model_file)
