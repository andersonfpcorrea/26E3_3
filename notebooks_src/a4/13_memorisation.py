from dlv.figures import plot_sample_rows
from dlv.gan import choose_by_mean_ratio

CHOICE_RULE = "versão com a maior razão média de diversidade na classe COVID-19 entre as três seeds"
choice = choose_by_mean_ratio(summaries, COVID_LABEL)
chosen_version = choice["chosen"]
print("regra:", CHOICE_RULE)
for key, value in choice["means"].items():
    print(f"{version_names[key]}: razão média {value:.3f}, desvio padrão {summaries[key]['diversity_ratio_std'][COVID_LABEL]:.3f}")
print("versão escolhida:", chosen_version, "-", version_names[chosen_version])

synthetic_count = train_counts["Normal"] - train_counts["COVID-19"]
synthetic_by_seed = {
    gan_seed: generate(
        generators[chosen_version][gan_seed], torch.full((synthetic_count,), COVID_LABEL), DEVICE, gan_seed
    )
    for gan_seed in GAN_SEEDS
}
synthetic_covid = torch.cat(list(synthetic_by_seed.values()))
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
    "seeds": list(GAN_SEEDS),
    "generated_to_train": describe_distances(generated_distance),
    "validation_to_train": describe_distances(real_distance),
    "generated_below_real_minimum": float((generated_distance < real_distance.min()).float().mean()),
}
print("sintéticas de COVID-19 (três seeds) até o treino:", memorisation["generated_to_train"])
print("validação real de COVID-19 até o treino:", memorisation["validation_to_train"])
print("parte das sintéticas mais perto do treino que a real mais próxima:", round(memorisation["generated_below_real_minimum"], 4))

sample_count = 8
samples_figure = plot_sample_rows(
    {
        "COVID-19 reais (treino)": train_covid[:sample_count],
        **{
            f"COVID-19 geradas, seed {gan_seed}": images[:sample_count]
            for gan_seed, images in synthetic_by_seed.items()
        },
    },
    run.figures_dir / "real_vs_synthetic.png",
)
