import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dlv.run import Stopwatch, seed_everything, start_run

SEED = 42
PER_CLASS = 2000
IMAGE_SIZE = 64
SCRATCH_BATCH_SIZE = 256
SCRATCH_EPOCHS = 40
SCRATCH_WARMUP_EPOCHS = 5
SCRATCH_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
PRETRAINED_BATCH_SIZE = 32
PRETRAINED_EPOCHS = 2
PRETRAINED_BACKBONE_LEARNING_RATE = 3e-5
PRETRAINED_HEAD_LEARNING_RATE = 1e-3
PRETRAINED_RESIZE = 256
PRETRAINED_CROP = 224
LOADER_WORKERS = 2
EXAMPLES_PER_OUTCOME = 3
MINIMUM_PER_STYLE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/wikiart")

seed_everything(SEED)
run = start_run("a1")
stopwatch = Stopwatch()

print("dispositivo:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
