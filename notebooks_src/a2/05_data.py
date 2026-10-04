from collections import Counter

from dlv.data import count_image_files, download_kaggle_dataset
from dlv.clip_search import list_ads16_images

with stopwatch.measure("download"):
    dataset_root = download_kaggle_dataset(DATASET_SLUG, DATA_DIR)

with stopwatch.measure("listagem"):
    images = list_ads16_images(dataset_root)
    files_found = count_image_files(dataset_root)

image_paths = [entry["path"] for entry in images]
sources = [entry["source"] for entry in images]
duplicates_removed = files_found - len(images)
thumbnail_files = sum(1 for entry in images if "_th_" in entry["path"].name)
by_source = dict(Counter(sources))
ad_categories = {
    str(category): count
    for category, count in sorted(
        Counter(entry["category"] for entry in images if entry["source"] == "anuncio").items()
    )
}
source_indices = {
    "corpus": list(range(len(images))),
    "anuncio": [i for i, source in enumerate(sources) if source == "anuncio"],
    "participante": [i for i, source in enumerate(sources) if source == "participante"],
}

print("arquivos de imagem no download:", files_found)
print("arquivos descartados (cópias exatas ou ilegíveis):", duplicates_removed)
print("imagens no corpus:", len(images))
print("arquivos com _th_ no nome (miniaturas de outra imagem):", thumbnail_files)
print("por origem:", by_source)
print("categorias de anúncio:", len(ad_categories))
print("anúncios por categoria:", ad_categories)
