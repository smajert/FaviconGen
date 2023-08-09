"""
Parameters used throughout the project
"""

from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"

DEVICE: str = "cuda"  # whether to run on GPU ("cuda") or CPU ("cpu")

EMBEDDING_DIM: int = 32  # dimension class labels and/or time step are transformed to
DO_NORM: bool = True  # whether to perform batch norm


class Dataset:  # everything related to MNIST/LLD
    specific_clusters: list[int] | None = [0, 1, 2]  # which LLD clusters to use; ignored for MNIST
    n_images: int | None = None  # total amount of images to load, None means all of them
    shuffle = True  # whether to shuffle the data


class AutoEncoder:  # everything related to VAE training
    adversarial_loss_weight: float | None = 1.0  # if given, use weighted patch discriminator as adversary;
    batch_size: int = 512
    epochs_mnist: int = 35  # epochs to train on MNIST
    epochs_lld: int = 350  # epochs to train on LLD
    kl_loss_weight: float = 1  # weight of Kullback Leibler vs. reconstruction loss for VAE
    learning_rate: float = 4e-4  # learning rate for VAE


class Diffusion:  # everything related to diffusion model training
    batch_size: int = 512
    epochs_mnist: int = 35  # epochs to train on MNIST
    epochs_lld: int = 250  # epochs to train on LLD
    guiding_factor: float = 0.90  # Guided (with label) vs. unguided (without labels) in classifier free guidance [4]
    learning_rate: float = 1e-3
    steps: int = 1000  # amount of time steps the diffusion model uses
    var_schedule_start: float = 0.0001  # starting value of the variance schedule beta
    var_schedule_end: float = 0.02  # final value (after `Diffusion.steps` time steps) of the variance schedule beta
