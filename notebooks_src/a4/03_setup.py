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

GAN_EPOCHS = 240
GAN_BATCH_SIZE = 64
GAN_BASE_CHANNELS = 32
GAN_SEEDS = (42, 43, 44)
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
}
EARLIER_EXECUTIONS = {
    "description": "Execuções anteriores de versões anteriores deste notebook, com uma seed por versão. Razão entre a diversidade gerada e a real na última décima parte das épocas. Não são resultados desta execução.",
    "classes": ["Normal", "Pneumonia", "COVID-19"],
    "executions": {
        "1 (64 canais)": {
            "v1 BCE": [0.98, 0.98, 0.66],
            "v2 normalização espectral e hinge": [0.07, 0.06, 0.04],
            "v3 WGAN-GP": None,
            "v4 BCE com aumento diferenciável": None,
        },
        "2 (32 canais)": {
            "v1 BCE": [0.96, 0.92, 0.56],
            "v2 normalização espectral e hinge": [0.05, 0.06, 0.04],
            "v3 WGAN-GP": [0.84, 0.82, 0.76],
            "v4 BCE com aumento diferenciável": None,
        },
        "3 (32 canais)": {
            "v1 BCE": [0.95, 0.80, 0.65],
            "v2 normalização espectral e hinge": [0.80, 0.41, 0.18],
            "v3 WGAN-GP": [0.90, 0.82, 0.91],
            "v4 BCE com aumento diferenciável": [0.82, 0.58, 0.45],
        },
    },
}

CLASSIFIER_EPOCHS = 10
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LEARNING_RATE = 1e-4
CLASSIFIER_INPUT_SIZE = 128
CLASSIFIER_SEEDS = GAN_SEEDS
MAX_SHIFT_PIXELS = 4.0
MAX_ROTATION_DEGREES = 7.0

CLASS_NAMES = [name for _, name in XRAY_CLASSES]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/xray")

seed_everything(SEED)
run = start_run("a4")
stopwatch = Stopwatch()

print("dispositivo:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
