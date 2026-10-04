from dlv.clip_search import (
    cached,
    embed_images,
    embed_texts,
    load_clip,
    similarity_matrix,
)

with stopwatch.measure("carregar CLIP"):
    model, processor = load_clip(DEVICE)

with stopwatch.measure("embeddings das imagens"):
    image_embeddings = cached(
        CACHE_PATH,
        lambda: embed_images(model, processor, image_paths, DEVICE, BATCH_SIZE),
        expected_rows=len(image_paths),
    )

with stopwatch.measure("embeddings dos textos"):
    concept_embeddings = embed_texts(model, processor, PROMPTS, DEVICE)
    query_embeddings = embed_texts(model, processor, QUERIES, DEVICE)

similarity = similarity_matrix(image_embeddings, concept_embeddings)
embedding_seconds = stopwatch.seconds["embeddings das imagens"]
print("embeddings das imagens:", tuple(image_embeddings.shape))
print("similaridade imagem x conceito:", tuple(similarity.shape))
print(f"similaridade mínima, média e máxima: {similarity.min():.3f}, {similarity.mean():.3f}, {similarity.max():.3f}")
