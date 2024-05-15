import json

from favicon_gen import params
from favicon_gen.denoising_diffusion import train as train_diffusion
from favicon_gen.autoencoder import train as train_autoencoder


if __name__ == "__main__":
    config = params.load_config()

    match config.model:
        case params.AutoEncoder():
            running_losses = train_autoencoder(config.dataset, config.model, config.general)
        case params.Diffusion():
            running_losses = train_diffusion(config.dataset, config.model, config.general)

    with open(params.OUTS_BASE_DIR / "loss.csv", "w", encoding="utf-8") as file:
        file.write("Epoch,Loss\n")
        for epoch, loss in enumerate(running_losses):
            file.write(f"{epoch},{loss}\n")

    with open(params.OUTS_BASE_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"final_loss": running_losses[-1]}, file)
