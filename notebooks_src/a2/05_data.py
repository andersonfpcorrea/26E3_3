from collections import Counter

from dlv.data import download_kaggle_dataset
from dlv.clip_search import scan_ads16

with stopwatch.measure("download"):
    dataset_root = download_kaggle_dataset(DATASET_SLUG, DATA_DIR)

with stopwatch.measure("listagem"):
    images, removal_counts = scan_ads16(dataset_root)

image_paths = [entry["path"] for entry in images]
sources = [entry["source"] for entry in images]
files_found = removal_counts["files_found"]
duplicates_removed = removal_counts["exact_copies"]
thumbnails_removed = removal_counts["thumbnails"]
unreadable_or_other = files_found - duplicates_removed - thumbnails_removed - len(images)
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
print("cópias exatas removidas:", duplicates_removed)
print("miniaturas removidas (a imagem original está no corpus):", thumbnails_removed)
print("arquivos ilegíveis:", unreadable_or_other)
print("imagens no corpus:", len(images))
print("por origem:", by_source)
print("categorias de anúncio:", len(ad_categories))
print("anúncios por categoria:", ad_categories)
