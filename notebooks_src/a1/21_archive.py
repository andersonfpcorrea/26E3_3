from dlv.run import save_run

metrics = {
    "periods": PERIOD_NAMES,
    "class_counts": class_counts,
    "split_sizes": split_sizes,
    "painters_per_split": painters_per_split,
    "unknown_painter_rows": unknown_painter_rows,
    "shared_painters": len(shared_painters),
    "rows_total": rows_total,
    "rows_with_year": rows_with_year,
    "rows_per_period": rows_per_period,
    "painter_check": painter_check,
    "largest_painter_per_period": largest_painter_per_period,
    "row_groups": row_groups,
    "peak_ram_after_reading_mb": peak_ram_after_reading_mb,
    "hyperparameters": {
        "seed": SEED,
        "per_class": PER_CLASS,
        "scratch": {
            "image_size": IMAGE_SIZE,
            "patch_size": 8,
            "embed_dim": 192,
            "depth": 6,
            "num_heads": 3,
            "batch_size": SCRATCH_BATCH_SIZE,
            "epochs": SCRATCH_EPOCHS,
            "warmup_epochs": SCRATCH_WARMUP_EPOCHS,
            "learning_rate": SCRATCH_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "schedule": "linear warm-up, cosine decay",
            "augmentation": "random crop with padding 4, horizontal flip",
            "channel_mean": channel_mean,
            "channel_std": channel_std,
        },
        "pretrained": {
            "backbone": "vit_b_16 IMAGENET1K_V1",
            "resize": PRETRAINED_RESIZE,
            "crop": PRETRAINED_CROP,
            "batch_size": PRETRAINED_BATCH_SIZE,
            "epochs": PRETRAINED_EPOCHS,
            "backbone_learning_rate": PRETRAINED_BACKBONE_LEARNING_RATE,
            "head_learning_rate": PRETRAINED_HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "mixed_precision": True,
        },
    },
    "models": summaries,
    "by_style": by_style,
    "attention_examples": attention_examples,
    "step_seconds": stopwatch.seconds,
}
archive = save_run(run, metrics)
saved = json.loads((run.output_dir / "metrics.json").read_text(encoding="utf-8"))
print("arquivo de saída:", archive)
print("tempo total (s):", saved["runtime_seconds"])
print("pico de memória de GPU (MB):", saved["peak_gpu_memory_mb"])
print("pico de memória RAM (MB):", saved["peak_ram_mb"])
