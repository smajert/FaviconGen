from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from tqdm import tqdm

from logo_maker.data_loading import load_logos
import logo_maker.params as params
from logo_maker.sample_from_model import sample_from_autoencoder_model

if __name__ == "__main__":
    batch_size = 256
    n_batches = 40
    model_sampler = sample_from_autoencoder_model(
        params.OUTS_BASE_DIR / "train_autoencoder_lld/model.pt",
        in_channels=3,
        seed=None,
        n_samples=batch_size,
        device="cuda",
    )

    to_normalized_image = transforms.Lambda(lambda t: (t + 1) / 2)
    inception = InceptionScore(normalize=True)
    inception.to("cuda")
    for _ in tqdm(range(n_batches)):
        gen_batch = to_normalized_image(next(model_sampler)).detach()
        inception.update(gen_batch)

    print(inception.compute())

