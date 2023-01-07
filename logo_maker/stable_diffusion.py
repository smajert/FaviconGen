from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    def __init__(self, beta_start: float = 0.0001, beta_end: float = 0.02, n_time_steps: int = 100, ):
        self.beta_schedule = torch.linspace(beta_start, beta_end, n_time_steps)
        alphas = 1 - self.beta_schedule
        alpha_cumulative_product = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_cumulative_product = torch.sqrt(alpha_cumulative_product)
        self.one_minus_sqrt_alpha_cumulative_product = 1 - self.sqrt_alpha_cumulative_product


SCHEDULE = NoiseSchedule()


def get_noisy_sample_at_step_t(
        original_image, t, device="cuda", noise_schedule: NoiseSchedule = SCHEDULE
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(original_image)
    noisy_sample = (
            original_image * noise_schedule.sqrt_alpha_cumulative_product[t]
            + noise * noise_schedule.one_minus_sqrt_alpha_cumulative_product[t]
    )

    return noisy_sample.to(device), noise.to(device)

