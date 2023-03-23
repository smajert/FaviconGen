from pathlib import Path

from logo_maker.data_loading import show_image_grid
from logo_maker.stable_diffusion import Generator, draw_sample_from_generator, NoiseSchedule
from matplotlib import pyplot as plt
import torch


def probe_model(model_file: Path, n_samples: int = 32, n_diffusion_steps: int = 300) -> None:
    model = Generator()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    schedule = NoiseSchedule(n_time_steps=n_diffusion_steps, linear_schedule_beta_start_end=(0.0001, 0.02))
    batch = draw_sample_from_generator(model, n_diffusion_steps, (n_samples, 3, 32, 32), schedule)
    show_image_grid(batch)
    plt.show()


if __name__ == "__main__":
    model_file = Path(r"C:\Users\steph\AppData\Local\Temp\logo_4wr962o0\model.pt")
    probe_model(model_file, n_diffusion_steps=1000)
