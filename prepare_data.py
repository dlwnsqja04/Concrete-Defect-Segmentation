import csv
import shutil
import sys
from pathlib import Path

import yaml

CLASSES = ("crack", "efflorescence", "spall_detach")
SPLITS = ("train", "val", "test")


def prepare(source: Path, manifest: Path, output: Path) -> None:
    """Restore the Week 2 split from its recorded list of included files."""
    with manifest.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["status"] == "used"]

    split_origins: dict[str, set[str]] = {split: set() for split in SPLITS}
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        split, name, origin = row["split"], row["image"], row["origin_id"]
        if (split not in split_origins or name != Path(name).name or not origin
                or (split, name) in seen_pairs):
            raise ValueError(f"Invalid manifest entry: {split}/{name}")
        if split != "train" and row["augmented"] != "False":
            raise ValueError(f"Unexpected holdout augmentation: {name}")
        seen_pairs.add((split, name))
        split_origins[split].add(origin)
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            overlap = split_origins[split] & split_origins[other]
            if overlap:
                raise ValueError(f"Origin overlap between {split} and {other}: {overlap}")

    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for row in rows:
        split, name = row["split"], row["image"]
        image = Path("images") / split / name
        label = Path("labels") / split / Path(name).with_suffix(".txt").name
        if not (source / image).stat().st_size:
            raise ValueError(f"Empty image in the restored split: {image}")
        shutil.copy2(source / image, output / image)
        shutil.copy2(source / label, output / label)
    config = {"path": str(output.resolve()), **{s: f"images/{s}" for s in SPLITS},
              "names": dict(enumerate(CLASSES))}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for split in SPLITS:
        print(split, sum(row["split"] == split for row in rows), len(split_origins[split]))


if __name__ == "__main__":
    prepare(*map(Path, sys.argv[1:4]))
