#!/usr/bin/env python3
"""Download a small, traceable sample from the public Gameplay_Images dataset."""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

DATASET = "Bingsu/Gameplay_Images"
GROUPS = {"genshin": (5000, 5), "terraria": (9000, 9)}


def get(url):
    for attempt in range(4):
        try:
            with urlopen(url, timeout=45) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 429:
                raise RuntimeError("Dataset server rate-limited the request; retry later") from exc
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets_raw"))
    parser.add_argument("--per-game", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.per_game <= 1000:
        parser.error("--per-game must be between 1 and 1000")
    for game, (base, expected_label) in GROUPS.items():
        directory = args.root / game
        directory.mkdir(parents=True, exist_ok=True)
        indices = [base + (i * 1000 // args.per_game) for i in range(args.per_game)]
        records = []
        for offset in range(base, base + 1000, 100):
            pending = [index for index in indices if offset <= index < offset + 100
                       and not (directory / f"hf_row_{index:05d}.png").exists()]
            rows = {}
            if pending:
                query = urlencode({"dataset": DATASET, "config": "default", "split": "train",
                                   "offset": offset, "length": 100})
                data = json.loads(get("https://datasets-server.huggingface.co/rows?" + query))
                rows = {entry["row_idx"]: entry["row"] for entry in data["rows"]}
            for index in [x for x in indices if offset <= x < offset + 100]:
                path = directory / f"hf_row_{index:05d}.png"
                if index in rows:
                    row = rows[index]
                    if row["label"] != expected_label:
                        raise RuntimeError(f"Unexpected label {row['label']} at row {index}")
                    path.write_bytes(get(row["image"]["src"]))
                records.append({"file": path.name, "dataset": DATASET, "split": "train",
                                "row_index": index, "class_label": expected_label})
                print(f"{game}: {len(records)}/{args.per_game}", flush=True)
        (directory / "source_manifest.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
