import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from dlv.run import Stopwatch, seed_everything, start_run
from dlv.xray import XRAY_CLASSES

SEED = 42
DATASET_SLUG = "tawsifurrahman/covid19-radiography-database"
INPUT_SIZE = 64
PROJECT_COUNTS = {0: 840, 1: 240, 2: 120}
TEST_PER_CLASS = 300
VALIDATION_SHARE = 0.2
NORMAL_LABEL = 0
COVID_LABEL = 2

ORIGINAL_EPOCHS = 15
ORIGINAL_BATCH_SIZE = 32
ORIGINAL_LEARNING_RATE = 0.01

GAN_EPOCHS = 300
GAN_BATCH_SIZE = 64
GAN_BASE_CHANNELS = 32
GAN_VERSIONS = {
    "v1": {
        "name": "Versão 1 (BCE)",
        "spectral_norm": False,
        "loss": "bce",
        "batch_norm": True,
        "gradient_penalty_weight": 0.0,
        "augment": False,
        "betas": (0.5, 0.999),
        "generator_lr": 2e-4,
        "discriminator_lr": 2e-4,
    },
    "v2": {
        "name": "Versão 2 (hinge, normalização espectral)",
        "spectral_norm": True,
        "loss": "hinge",
        "batch_norm": False,
        "gradient_penalty_weight": 0.0,
        "augment": False,
        "betas": (0.5, 0.999),
        "generator_lr": 1e-4,
        "discriminator_lr": 4e-4,
    },
    "v3": {
        "name": "Versão 3 (WGAN-GP)",
        "spectral_norm": False,
        "loss": "wgan",
        "batch_norm": False,
        "gradient_penalty_weight": 10.0,
        "augment": False,
        "betas": (0.0, 0.9),
        "generator_lr": 1e-4,
        "discriminator_lr": 1e-4,
    },
    "v4": {
        "name": "Versão 4 (BCE com aumento diferenciável)",
        "spectral_norm": False,
        "loss": "bce",
        "batch_norm": True,
        "gradient_penalty_weight": 0.0,
        "augment": True,
        "betas": (0.5, 0.999),
        "generator_lr": 2e-4,
        "discriminator_lr": 2e-4,
    },
}

CLASSIFIER_EPOCHS = 10
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LEARNING_RATE = 1e-4
CLASSIFIER_INPUT_SIZE = 128
CLASSIFIER_SEEDS = (42, 43, 44)
MAX_SHIFT_PIXELS = 4.0
MAX_ROTATION_DEGREES = 7.0

CLASS_NAMES = [name for _, name in XRAY_CLASSES]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/xray")

seed_everything(SEED)
run = start_run("a4")
stopwatch = Stopwatch()

print("dispositivo:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
