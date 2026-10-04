from dlv.clip_search import concept_probabilities, sample_in_band
from dlv.figures import plot_image_grid, plot_probability_distributions

probabilities = concept_probabilities(similarity, LOGIT_SCALE)
top_probability, top_concept = probabilities.max(dim=1)
distribution_figure = plot_probability_distributions(
    top_probability, probabilities, THRESHOLD, run.figures_dir / "distribution.png"
)

probability_quantiles = {
    str(level): round(top_probability.quantile(level).item(), 3) for level in (0.05, 0.25, 0.5, 0.75, 0.95)
}
print("quantis da probabilidade mais alta por imagem:", probability_quantiles)
share_above = {}
for candidate in CANDIDATE_THRESHOLDS:
    share_above[str(candidate)] = round((top_probability >= candidate).float().mean().item(), 4)
    print(f"limiar {candidate}: {share_above[str(candidate)]:.1%} das imagens têm algum conceito com probabilidade igual ou maior")

threshold_samples = {}
threshold_figures = []
for candidate in CANDIDATE_THRESHOLDS:
    below = sample_in_band(top_probability, candidate - BAND_HALF_WIDTH, candidate, SAMPLES_PER_SIDE, SEED)
    above = sample_in_band(top_probability, candidate, candidate + BAND_HALF_WIDTH, SAMPLES_PER_SIDE, SEED)
    chosen = below + above
    threshold_samples[str(candidate)] = {"below": below, "above": above}
    threshold_figures.append(
        plot_image_grid(
            [image_paths[i] for i in chosen],
            [f"{CONCEPTS[top_concept[i]]}\n{top_probability[i]:.2f} {sources[i]}" for i in chosen],
            f"Limiar {candidate}: acima (linha de baixo) e abaixo (linha de cima)",
            run.figures_dir / f"threshold_{int(candidate * 100):03d}.png",
            columns=SAMPLES_PER_SIDE,
        )
    )
    print(f"limiar {candidate}: abaixo {below}, acima {above}")
