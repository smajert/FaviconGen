"""
Parameters used throughout the project
"""
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any
from pathlib import Path

from dacite import Config, from_dict
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"


class AvailableDatasets(Enum):
    LLD = auto()  # Large Logo Dataset
    MNIST = auto()  # Modified National Institute of Standards and Technology database


@dataclass
class Dataset:  # everything related to MNIST/LLD
    name: AvailableDatasets
    n_images: int | None  # total amount of images to load, None means all of them
    shuffle: bool  # whether to shuffle the data
    specific_clusters: list[int] | None = None  # which LLD clusters to use

    @property
    def in_channels(self):
        """Get number of channels, i.e. 3 for colored images and 1 for greyscale"""
        match self.name:
            case AvailableDatasets.LLD:
                return 3
            case AvailableDatasets.MNIST:
                return 1


@dataclass
class General:  # Parameters relevant for all models
    device: str  # whether to run on GPU ("cuda") or CPU ("cpu")
    do_norm: bool  # whether to perform batch norm
    embedding_dim: int  # dimension time steps are transformed to
    batch_size: int
    epochs: int
    learning_rate: float
    lr_reduction_patience: int  # patience for reduction of learning rate in epochs


@dataclass
class AutoEncoder:  # everything related to VAE training
    adversarial_loss_weight: float | None
    kl_loss_weight: float  # how strongly to force latent space to gaussian distribution


class DiffusionArchitecture(Enum):
    CUSTOM = "CUSTOM"
    UNET2D = "UNET2D"


@dataclass
class Diffusion:  # everything related to diffusion model training
    architecture: DiffusionArchitecture
    steps: int  # amount of time steps the diffusion model uses
    var_schedule_start: float  # starting value of the variance schedule beta
    var_schedule_end: float  # final value (after `steps` time steps) of the variance schedule beta


@dataclass
class ProjectConfig:
    dataset: Dataset
    general: General
    model: AutoEncoder | Diffusion


def load_config() -> ProjectConfig:
    @dataclass
    class DummyProjectConfig:
        general: General
        dataset: Dataset
        model: Any  # currently, Omegaconf cannot deal with the union container here -> dummy class

    schema = OmegaConf.structured(DummyProjectConfig)

    cfg = OmegaConf.merge(schema, OmegaConf.load("params.yaml"))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return from_dict(
        data_class=ProjectConfig, data=cfg_dict, config=Config(strict=True, cast=[Enum])
    )
