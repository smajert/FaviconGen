from favicon_gen import params
from favicon_gen.denoising_diffusion import train as train_diffusion
from favicon_gen.autoencoder import train as train_autoencoder


if __name__ == "__main__":
    config = params.load_config()

    match config.model:
        case params.AutoEncoder():
            train_autoencoder(config.dataset, config.model, config.general)
        case params.Diffusion():
            train_diffusion(config.dataset, config.model, config.general)
