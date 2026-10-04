from dlv.clip_search import search
from dlv.figures import plot_image_grid

queries = []
query_figures = []
for number, (query, embedding) in enumerate(zip(QUERIES, query_embeddings), start=1):
    with stopwatch.measure(f"busca {number}"):
        found = search(embedding, image_embeddings, RESULTS_PER_QUERY)
    chosen = [index for index, _ in found]
    queries.append(
        {
            "text": query,
            "results": [
                {"index": index, "score": round(score, 4), "source": sources[index], "file": image_paths[index].name}
                for index, score in found
            ],
        }
    )
    query_figures.append(
        plot_image_grid(
            [image_paths[index] for index in chosen],
            [f"{score:.3f} {sources[index]}" for index, score in found],
            query,
            run.figures_dir / f"query_{number:02d}.png",
            columns=RESULTS_PER_QUERY,
        )
    )
    print(f"{query!r}:", [(row["index"], row["score"], row["source"]) for row in queries[-1]["results"]])
