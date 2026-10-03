from dlv.run import save_run

metrics = {
    "class_names": class_names,
    "class_counts": class_counts,
    "split_sizes": split_sizes,
    "files_found": files_found,
    "duplicates_removed": duplicates_removed,
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
}
archive = save_run(run, metrics)
saved = json.loads((run.output_dir / "metrics.json").read_text(encoding="utf-8"))
print("arquivo de saída:", archive)
print("tempo total (s):", saved["runtime_seconds"])
print("pico de memória de GPU (MB):", saved["peak_gpu_memory_mb"])
print("pico de memória RAM (MB):", saved["peak_ram_mb"])
