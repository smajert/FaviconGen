from matplotlib import pyplot as plt
import numpy as np

import logo_maker.params as params
from logo_maker.denoising_diffusion import VarianceSchedule

if __name__ == "__main__":
    schedule = VarianceSchedule(
        beta_start_end=(params.Diffusion.VAR_SCHEDULE_START, params.Diffusion.VAR_SCHEDULE_END),
        n_time_steps=params.Diffusion.DIFFUSION_STEPS
    )
    factor = schedule.beta_t / (1 - schedule.alpha_bar_t)
    epsilon_r = np.sqrt(schedule.beta_t / (schedule.alpha_t * (1 - schedule.alpha_bar_t)))
    alpha_bar_t_shifted = np.roll(schedule.alpha_bar_t, shift=1)
    epsilon_r_naive = np.sqrt((1 - schedule.alpha_bar_t) / (schedule.alpha_bar_t * (1 - alpha_bar_t_shifted)))
    gamma = np.sqrt(schedule.beta_t)
    gamma_naive = np.sqrt(1 - alpha_bar_t_shifted)

    plt.figure()
    plt.plot(epsilon_r, label=r"$\epsilon_r$")
    plt.plot(epsilon_r_naive, label=r"$\epsilon_{r,naive}$")
    plt.plot(gamma, label=r"$\gamma$")
    plt.plot(gamma_naive, label=r"$\gamma_{naive}$")
    plt.legend()
    plt.grid()
    plt.show()


