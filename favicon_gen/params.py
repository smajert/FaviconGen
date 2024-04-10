"""
Parameters used throughout the project
"""
from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"


@dataclass
class General:
    device: str # whether to run on GPU ("cuda") or CPU ("cpu")
    embedding_dim: int  # dimension class labels and/or time step are transformed to
    do_norm: bool # whether to perform batch norm


class AvailableDatasets(Enum):
    LLD = auto()
    MNIST = auto()


@dataclass
class Dataset:  # everything related to MNIST/LLD
    name: AvailableDatasets
    n_images: Optional[int]  # total amount of images to load, None means all of them
    shuffle: bool  # whether to shuffle the data
    specific_clusters: Optional[list[int]] = None  # which LLD clusters to use


@dataclass
class AutoEncoder:  # everything related to VAE training
    adversarial_loss_weight: Optional[float]
    batch_size: int
    epochs_mnist: int  # epochs to train on MNIST
    epochs_lld: int  # epochs to train on LLD
    kl_loss_weight: float  # weight of Kullback Leibler vs. reconstruction loss for VAE
    learning_rate: float  # learning rate for VAE


@dataclass
class Diffusion:  # everything related to diffusion model training
    batch_size: int
    epochs_mnist: int  # epochs to train on MNIST
    epochs_lld: int  # epochs to train on LLD
    guiding_factor: float  # Guided (with label) vs. unguided (without labels) in classifier free guidance [4]
    learning_rate: float
    steps: int # amount of time steps the diffusion model uses
    var_schedule_start: float # starting value of the variance schedule beta
    var_schedule_end: float  # final value (after `Diffusion.steps` time steps) of the variance schedule beta


@dataclass
class ProjectConfig:
    general: General
    dataset: Dataset
    autoencoder: AutoEncoder
    diffusion: Diffusion


def load_config() -> ProjectConfig:
    schema = OmegaConf.structured(ProjectConfig)
    cfg = OmegaConf.merge(schema, OmegaConf.load("params.yaml"))
    return cfg
