import json
import random
import time
from pathlib import Path
from shutil import make_archive

import numpy as np
import torch
from torch.utils.data import DataLoader

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DATASET_SLUG = "pavansanagapati/images-dataset"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/images_dataset")
OUTPUT_DIR = Path("outputs/a3")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STARTED_AT = time.time()

print("dispositivo:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
