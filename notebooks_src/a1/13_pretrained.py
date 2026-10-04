from dlv.data import imagenet_transform
from dlv.paintings import EncodedImageDataset
from dlv.vit import build_pretrained_vit, fine_tuning_groups

pretrained_transform = imagenet_transform(PRETRAINED_RESIZE, PRETRAINED_CROP)


def make_loader(indices, shuffle):
    """Builds a loader that decodes the 256x256 images selected by the given indices."""
    dataset = EncodedImageDataset(
        [encoded_images[index] for index in indices],
        [labels[index] for index in indices],
        pretrained_transform,
    )
    return DataLoader(
        dataset,
        batch_size=PRETRAINED_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=LOADER_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )


PRETRAINED_NAME = "ViT-B/16 pré-treinado"
seed_everything(SEED)
pretrained_model = build_pretrained_vit(num_classes=len(PERIOD_NAMES)).to(DEVICE)
pretrained_optimizer = torch.optim.AdamW(
    fine_tuning_groups(
        pretrained_model, PRETRAINED_BACKBONE_LEARNING_RATE, PRETRAINED_HEAD_LEARNING_RATE
    ),
    weight_decay=WEIGHT_DECAY,
)
with stopwatch.measure(f"treino, {PRETRAINED_NAME}"):
    pretrained_history = fit(
        pretrained_model,
        make_loader(train_indices, shuffle=True),
        make_loader(validation_indices, shuffle=False),
        pretrained_optimizer,
        epochs=PRETRAINED_EPOCHS,
        device=DEVICE,
        use_amp=True,
    )
with stopwatch.measure(f"avaliação, {PRETRAINED_NAME}"):
    pretrained_report = evaluate(
        pretrained_model,
        make_loader(test_indices, shuffle=False),
        DEVICE,
        num_classes=len(PERIOD_NAMES),
        use_amp=True,
    )
pretrained_summary = summarize(
    pretrained_model,
    pretrained_history,
    pretrained_report,
    stopwatch.seconds[f"treino, {PRETRAINED_NAME}"],
)
print("parâmetros:", pretrained_summary["parameters"])
print(f"acurácia no teste: {pretrained_summary['accuracy']:.4f}")
