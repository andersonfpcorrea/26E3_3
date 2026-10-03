peak_memory = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == "cuda" else 0.0
metrics = {
    "class_names": class_names,
    "class_counts": class_counts,
    "split_sizes": split_sizes,
    "hyperparameters": {
        "backbone": "resnet50 IMAGENET1K_V2",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "seed": SEED,
    },
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_count,
    "history": history,
    "accuracy": report["accuracy"],
    "per_class_accuracy": dict(zip(class_names, report["per_class_accuracy"])),
    "confusion": report["confusion"],
    "runtime_seconds": round(time.time() - STARTED_AT, 1),
    "peak_gpu_memory_mb": round(peak_memory, 1),
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(make_archive("outputs_a3", "zip", OUTPUT_DIR))
print("tempo total (s):", metrics["runtime_seconds"])
print("pico de memória de GPU (MB):", metrics["peak_gpu_memory_mb"])
