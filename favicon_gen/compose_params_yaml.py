"""
Explicitly compose a params.yaml from
the hydra config files stored in the 'conf'
folder. 
For some reason DVC does not come with this 
functionality outside of using 'dvc exp run'.
"""

from hydra import compose, initialize
from omegaconf import OmegaConf

from favicon_gen.params import REPO_ROOT


if __name__ == "__main__":
    # context initialization
    with initialize(version_base=None, config_path="../conf", job_name="test_app"):
        cfg = compose(config_name="config")

    with open(REPO_ROOT / "params.yaml", "w", encoding="utf-8") as params_yaml:
        OmegaConf.save(config=cfg, f=params_yaml.name)
