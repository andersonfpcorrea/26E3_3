import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dlv.run import seed_everything, start_run

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DATASET_SLUG = "pavansanagapati/images-dataset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/images_dataset")

seed_everything(SEED)
run = start_run("a3")

print("dispositivo:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
