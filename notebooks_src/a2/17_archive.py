from dlv.run import save_run

metrics = {
    "files_found": files_found,
    "duplicates_removed": duplicates_removed,
    "thumbnails_removed": thumbnails_removed,
    "images": len(images),
    "by_source": by_source,
    "ad_categories": ad_categories,
    "concepts": PROMPTS,
    "threshold": {
        "value": THRESHOLD,
        "rule": "um conceito está presente quando a sua probabilidade na softmax entre os conceitos é igual ou maior que o valor",
        "candidates": CANDIDATE_THRESHOLDS,
        "share_of_images_above": share_above,
        "top_probability_quantiles": probability_quantiles,
        "samples": threshold_samples,
    },
    "ranking": ranking,
    "top_examples": top_examples,
    "queries": queries,
    "tokenisation": tokenisation,
    "embedding_seconds": embedding_seconds,
    "step_seconds": stopwatch.seconds,
    "model": "openai/clip-vit-base-patch32",
    "logit_scale": LOGIT_SCALE,
    "seed": SEED,
}
archive = save_run(run, metrics)
saved = json.loads((run.output_dir / "metrics.json").read_text(encoding="utf-8"))
print("arquivo de saída:", archive)
print("tempo total (s):", saved["runtime_seconds"])
print("pico de memória de GPU (MB):", saved["peak_gpu_memory_mb"])
print("pico de memória RAM (MB):", saved["peak_ram_mb"])
