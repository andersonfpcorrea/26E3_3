from dlv.paintings import row_group_sizes, spelling_variants
from dlv.run import peak_ram_mb

known_artist_counts = Counter(artist for artist in artists if not is_unknown_artist(artist))
variants_in_sample = spelling_variants(known_artist_counts)
variants_in_dataset = spelling_variants(
    {artist for artist in metadata["artist"] if not is_unknown_artist(artist)}
)
split_of_painter = {
    artist: split_name for split_name, names in zip(SPLIT_NAMES, known_painters) for artist in names
}
largest_painter_per_period = {}
for label, name in enumerate(PERIOD_NAMES):
    painter, count = Counter(
        artist
        for artist, painting_label in zip(artists, labels)
        if painting_label == label and not is_unknown_artist(artist)
    ).most_common(1)[0]
    largest_painter_per_period[name] = {
        "painter": painter,
        "paintings": count,
        "split": split_of_painter[painter],
    }
row_groups = [
    {"file": path.name, "groups": len(sizes), "smallest": min(sizes), "largest": max(sizes)}
    for path, sizes in zip(parquet_paths, row_group_sizes(parquet_paths))
]
painter_check = {
    "distinct_painters": len(known_artist_counts),
    "unknown_painter_rows": sum(unknown_painter_rows.values()),
    "most_frequent": dict(known_artist_counts.most_common(10)),
    "spelling_variants_in_sample": variants_in_sample,
    "spelling_variants_in_dataset": variants_in_dataset,
}
peak_ram_after_reading_mb = round(peak_ram_mb(), 1)

print("pintores distintos na amostra:", painter_check["distinct_painters"])
print("pinturas sem pintor conhecido na amostra:", painter_check["unknown_painter_rows"])
print("dez pintores mais frequentes na amostra:", painter_check["most_frequent"])
print("pintores com mais de uma grafia na amostra:", len(variants_in_sample), variants_in_sample[:10])
print("pintores com mais de uma grafia no dataset inteiro:", len(variants_in_dataset), variants_in_dataset[:10])
for name, entry in largest_painter_per_period.items():
    print(f"maior pintor em {name}: {entry['painter']}, {entry['paintings']} pinturas, {entry['split']}")
for entry in row_groups:
    print(
        f"{entry['file']}: {entry['groups']} blocos, de {entry['smallest']} a {entry['largest']} linhas"
    )
print("pico de memória RAM até aqui (MB):", peak_ram_after_reading_mb)
