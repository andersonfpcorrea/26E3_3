import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dlv.run import Stopwatch, seed_everything, start_run

SEED = 42
DATASET_SLUG = "groffo/ads16-dataset"
DATA_DIR = Path("data/ads16")
CACHE_PATH = Path("cache/a2_image_embeddings.pt")
BATCH_SIZE = 64
LOGIT_SCALE = 100.0
THRESHOLD = 0.5
CANDIDATE_THRESHOLDS = [0.3, 0.5, 0.7]
SAMPLES_PER_SIDE = 8
BAND_HALF_WIDTH = 0.05
TOP_CONCEPTS = 5
EXAMPLES_PER_CONCEPT = 6
RESULTS_PER_QUERY = 5

CONCEPTS = [
    "a car",
    "a person",
    "a smiling woman",
    "a man",
    "a child",
    "food",
    "a drink",
    "a mobile phone",
    "a computer",
    "clothes",
    "shoes",
    "jewellery",
    "cosmetics",
    "furniture",
    "a house",
    "outdoor scenery",
    "a beach",
    "an animal",
    "sports",
    "a city street",
    "a musical instrument",
    "text and a logo",
    "a product on a plain background",
    "an illustration or cartoon",
    "a chart or diagram",
    "an online advertisement with a lot of text",
]
PROMPTS = [f"a photo of {concept}" for concept in CONCEPTS]

QUERIES = [
    "a car",
    "a red sports car on a road",
    "a happy family at home",
    "luxury",
    "freedom",
    "something healthy to eat",
    "technology that makes life easier",
    "an advertisement with a lot of text",
    "loneliness",
    "um cachorro brincando na praia",
    "uma pessoa sorrindo",
]
TOKEN_TEXTS = [
    "a car",
    "a red sports car on a road",
    "a happy family sitting together at the kitchen table at home, "
    "eating breakfast and laughing while the morning sun comes in through the window",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(SEED)
run = start_run("a2")
stopwatch = Stopwatch()

print("dispositivo:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
