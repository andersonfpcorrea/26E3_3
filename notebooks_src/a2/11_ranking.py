from dlv.clip_search import rank_concepts, top_images
from dlv.figures import plot_image_grid, plot_ranking

ranking = {}
for name, indices in source_indices.items():
    ranking[name] = rank_concepts(similarity[indices], PROMPTS, THRESHOLD, LOGIT_SCALE)
    print(f"ranking: {name} ({len(indices)} imagens), limiar {THRESHOLD}")
    for position, row in enumerate(ranking[name], start=1):
        print(
            f"{position:2d}. {row['concept']}: frequência {row['frequency']:.3f}, "
            f"contagem {row['count']}, similaridade média {row['mean_similarity']:.3f}"
        )
ranking_figure = plot_ranking(ranking, run.figures_dir / "ranking.png")

top_examples = {}
top_example_figures = []
for row in ranking["corpus"][:TOP_CONCEPTS]:
    concept_index = PROMPTS.index(row["concept"])
    chosen = top_images(similarity, concept_index, EXAMPLES_PER_CONCEPT)
    top_examples[row["concept"]] = {
        "indices": chosen,
        "scores": [round(similarity[i, concept_index].item(), 4) for i in chosen],
        "sources": [sources[i] for i in chosen],
    }
    top_example_figures.append(
        plot_image_grid(
            [image_paths[i] for i in chosen],
            [f"{similarity[i, concept_index]:.3f} {sources[i]}" for i in chosen],
            row["concept"],
            run.figures_dir / f"top_{len(top_example_figures) + 1}.png",
            columns=EXAMPLES_PER_CONCEPT,
        )
    )
