from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"


class AutoEncoderParams:
    N_EPOCHS = 150
    BATCH_SIZE = 128
