from dlv.figures import plot_sample_rows

chosen_version = min(
    GAN_VERSIONS,
    key=lambda key: sum(abs(ratio - 1) for ratio in diagnosis[key]["diversity_ratio"].values()) / len(CLASS_NAMES),
)
print("versão escolhida:", chosen_version, "-", version_names[chosen_version])

synthetic_count = train_counts["Normal"] - train_counts["COVID-19"]
synthetic_covid = generate(
    generators[chosen_version], torch.full((synthetic_count,), COVID_LABEL), DEVICE, SEED
)
train_covid = train_images[train_labels == COVID_LABEL]
validation_covid = validation_images[validation_labels == COVID_LABEL]
generated_distance = nearest_real_distance(synthetic_covid, train_covid)
real_distance = nearest_real_distance(validation_covid, train_covid)


def describe_distances(distances):
    """Summarises nearest-image distances by mean, median and minimum."""
    return {
        "count": int(distances.numel()),
        "mean": float(distances.mean()),
        "median": float(distances.median()),
        "minimum": float(distances.min()),
    }


memorisation = {
    "chosen_version": chosen_version,
    "generated_to_train": describe_distances(generated_distance),
    "validation_to_train": describe_distances(real_distance),
    "generated_below_real_minimum": float((generated_distance < real_distance.min()).float().mean()),
}
print("sintéticas de COVID-19 até o treino:", memorisation["generated_to_train"])
print("validação real de COVID-19 até o treino:", memorisation["validation_to_train"])
print("parte das sintéticas mais perto do treino que a real mais próxima:", round(memorisation["generated_below_real_minimum"], 4))

sample_count = 8
sample_labels = torch.full((sample_count,), COVID_LABEL)
samples_figure = plot_sample_rows(
    {
        "COVID-19 reais (treino)": train_covid[:sample_count],
        **{
            f"COVID-19 geradas, {version_names[key]}": generate(generators[key], sample_labels, DEVICE, SEED)
            for key in GAN_VERSIONS
        },
    },
    run.figures_dir / "real_vs_synthetic.png",
)
