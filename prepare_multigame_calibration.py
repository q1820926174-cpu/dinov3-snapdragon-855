#!/usr/bin/env python3
"""Small, reproducible, disjoint multi-game calibration/evaluation split.

Inputs are local public dataset exports organized under --root/{game}/...
This script DOES NOT download/redistribute commercial game screenshots.
Calibration: 24 images per game; held-out evaluation: 8 per game.
DINOv3 inputs are float32 .raw files for SNPE's static activation calibration.

The preprocessing used here (square letterbox + DINOv3 normalization) must be
identical for both FP32 baseline and mobile deployment, otherwise comparisons
are invalid. Supports --dry-run to validate sources/splits without a model.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

GAMES = ("pvz", "hollow_knight", "terraria", "cuphead", "clash_royale", "genshin")
MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Candidate:
    path: Path
    parent_split: str
    family: str
    video_id: str | None = None


def original_family(path: Path) -> str:
    # Roboflow files often have generated suffixes like: foo_jpg.rf.UUID.jpg
    s = path.stem.lower().split(".rf.")[0]
    # Keep Unicode names: stripping them makes unrelated non-ASCII files collide.
    return re.sub(r"[^\w-]", "", s) or path.stem.lower()


def split_hint(path: Path) -> str:
    names = {x.lower() for x in path.parts}
    if "test" in names:
        return "test"
    if "valid" in names or "val" in names or "validation" in names:
        return "valid"
    if "train" in names:
        return "train"
    return "unspecified"


def dhash(img: Image.Image) -> int:
    # 64-bit difference hash for cheap nearby-frame duplicate screening.
    gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = list(gray.get_flattened_data()) if hasattr(gray, "get_flattened_data") else list(gray.getdata())
    h = 0
    for y in range(8):
        for x in range(8):
            h = (h << 1) | int(values[y * 9 + x] > values[y * 9 + x + 1])
    return h


def candidates(root: Path) -> list[Candidate]:
    video_ids = {}
    for manifest in root.glob('source_manifest*.json'):
        for row in json.loads(manifest.read_text(encoding='utf-8')):
            video = row.get('source_video_id') or row.get('video_id')
            if video:
                video_ids[(root / row['file']).resolve()] = str(video)
    return [Candidate(path=p, parent_split=split_hint(p.relative_to(root)), family=original_family(p),
                      video_id=video_ids.get(p.resolve()))
            for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXT]


def pick(items: list[Candidate], count: int, rng: random.Random, families: set[str],
         hashes: list[int], near_duplicate: int) -> list[tuple[Candidate, int]]:
    rng.shuffle(items)
    selected = []
    for item in items:
        if item.family in families:
            continue
        try:
            with Image.open(item.path) as image:
                value = dhash(ImageOps.exif_transpose(image))
        except (UnidentifiedImageError, OSError):
            continue
        if any((value ^ old).bit_count() <= near_duplicate for old in hashes):
            continue
        selected.append((item, value))
        families.add(item.family)
        hashes.append(value)
        if len(selected) == count:
            break
    return selected


def get_yolo_label(image: Path) -> Path | None:
    # Roboflow exports: split/images/foo.jpg -> split/labels/foo.txt.
    if image.parent.name.lower() != "images":
        return None
    label = image.parent.parent / "labels" / (image.stem + ".txt")
    return label if label.is_file() else None


def calibration_roi(image: Image.Image, ordinal: int, rng: random.Random) -> tuple[Image.Image, str]:
    # 24 inputs: 12 full, 8 center, 4 corner/UI; no image replicated.
    mode = ["full", "full", "full", "center", "center", "ui"][ordinal % 6]
    w, h = image.size
    if mode == "full":
        return image, mode
    side = min(w, h)
    if mode == "center":
        x, y = (w - side) // 2, (h - side) // 2
        return image.crop((x, y, x + side, y + side)), mode
    # Corners are candidate HUD/UI areas, not guaranteed actual UI labels.
    crop_size = max(16, side // 2)
    x = rng.choice((0, w - crop_size))
    y = rng.choice((0, h - crop_size))
    return image.crop((x, y, x + crop_size, y + crop_size)), mode


def to_f32_raw(image: Image.Image, dst: Path, processor, image_size: int, layout: str) -> list[int]:
    import numpy as np

    # Deliberate preprocess contract: pad full frame or ROI, then normalize.
    # This must also be used by the FP32 baseline and on-device pipeline.
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    square = ImageOps.pad(rgb, (image_size, image_size), method=Image.Resampling.BICUBIC,
                          color=(114, 114, 114), centering=(0.5, 0.5))
    # DINOv3's fast image processor only returns PyTorch tensors.
    tensor = processor(images=square, do_resize=False, do_center_crop=False,
                       return_tensors="pt")["pixel_values"].cpu().numpy().astype(np.float32)
    if tensor.shape != (1, 3, image_size, image_size):
        raise RuntimeError(f"Unexpected preprocessor output {tensor.shape}; check the processor version")
    if layout == "NHWC":
        tensor = np.transpose(tensor, (0, 2, 3, 1))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(tensor).tofile(dst)
    return list(tensor.shape)


def main() -> None:
    p = argparse.ArgumentParser(description="144 calibration + 48 hold-out screenshots by default")
    p.add_argument("--root", type=Path, default=Path("datasets_raw"))
    p.add_argument("--out", type=Path, default=Path("small_game_set"))
    p.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES),
                   help="Select available games for a partial smoke test")
    p.add_argument("--model", default=MODEL, help="Hugging Face model ID or local processor directory")
    p.add_argument("--calib-per-game", type=int, default=24)
    p.add_argument("--test-per-game", type=int, default=8)
    p.add_argument("--size", type=int, choices=(224, 320, 448), default=224)
    p.add_argument("--layout", choices=("NCHW", "NHWC"), required=True,
                   help="Choose ONLY after checking SNPE DLC input format")
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--near-duplicate", type=int, default=5, help="dHash Hamming distance threshold")
    p.add_argument("--dry-run", action="store_true", help="Check splits; don't load model or generate .raw")
    args = p.parse_args()
    if args.calib_per_game < 1 or args.test_per_game < 1:
        p.error("sample counts must be >= 1")
    if not 0 <= args.near_duplicate <= 64:
        p.error("--near-duplicate must be between 0 and 64")
    if len(set(args.games)) != len(args.games):
        p.error("--games must not contain duplicates")

    missing = [g for g in args.games if not (args.root / g).is_dir()]
    if missing:
        p.error(f"Missing dataset folders: {', '.join(missing)} under {args.root.resolve()}")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    processor = None
    if not args.dry_run:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(args.model)

    manifest = []
    cal_lines, test_lines = [], []
    counts = {}
    for group in args.games:
        source = args.root / group
        all_items = candidates(source)
        families, hashes = set(), []
        # Prefer downloaded official dataset test/valid split for held-out images.
        eval_first = [item for item in all_items if item.parent_split in ("valid", "test")]
        eval_rest = [item for item in all_items if item.parent_split not in ("valid", "test")]
        eval_selected = pick(eval_first, args.test_per_game, rng, families, hashes, args.near_duplicate)
        if len(eval_selected) < args.test_per_game:
            eval_selected += pick(eval_rest, args.test_per_game - len(eval_selected), rng,
                                  families, hashes, args.near_duplicate)
        held_out_videos = {item.video_id for item, _ in eval_selected if item.video_id}
        cal_first = [item for item in all_items if item.parent_split in ("train", "unspecified")
                     and item.video_id not in held_out_videos]
        cal_rest = [item for item in all_items if item.parent_split in ("test", "valid")
                    and item.video_id not in held_out_videos]
        cal_selected = pick(cal_first, args.calib_per_game, rng, families, hashes, args.near_duplicate)
        if len(cal_selected) < args.calib_per_game:
            cal_selected += pick(cal_rest, args.calib_per_game - len(cal_selected), rng,
                                 families, hashes, args.near_duplicate)
        if len(eval_selected) < args.test_per_game or len(cal_selected) < args.calib_per_game:
            raise RuntimeError(f"{group} insufficient distinct frames: got {len(cal_selected)} calib and "
                               f"{len(eval_selected)} test. Remove near-duplicates or lower counts.")
        counts[group] = {"calibration": len(cal_selected), "test": len(eval_selected)}
        for split, chosen in (("calibration", cal_selected), ("test", eval_selected)):
            for i, (item, h) in enumerate(chosen):
                with Image.open(item.path) as img:
                    original = ImageOps.exif_transpose(img).convert("RGB")
                    roi, mode = calibration_roi(original, i, rng) if split == "calibration" else (original, "full")
                    name = f"{group}_{i:03d}"
                    raw_relpath = f"{split}_raw/{name}.raw"
                    shape = None
                    if processor is not None:
                        shape = to_f32_raw(roi, args.out / raw_relpath, processor, args.size, args.layout)
                        if split == "calibration":
                            cal_lines.append(str((args.out / raw_relpath).resolve()))
                        else:
                            test_lines.append(str((args.out / raw_relpath).resolve()))
                    if split == "test":
                        png = args.out / "test_images" / group / f"{name}.png"
                        png.parent.mkdir(parents=True, exist_ok=True)
                        original.save(png)
                        label = get_yolo_label(item.path)
                        if label:
                            outlabel = args.out / "test_labels" / group / f"{name}.txt"
                            outlabel.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(label, outlabel)
                    manifest.append({
                        "game": group,
                        "split": split,
                        "source": str(item.path.resolve()),
                        "source_family": item.family,
                        "source_video_id": item.video_id,
                        "original_split": item.parent_split,
                        "original_size": list(original.size),
                        "roi": mode,
                        "dhash64": f"{h:016x}",
                        "raw_file": raw_relpath if processor is not None else None,
                        "shape": shape,
                        "layout": args.layout,
                        "seed": args.seed,
                    })

    (args.out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in manifest) + "\n",
        encoding="utf-8",
    )
    if processor is not None:
        (args.out / "calibration_list.txt").write_text("\n".join(cal_lines) + "\n", encoding="utf-8")
        (args.out / "evaluation_list.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps({
        "calibration_total": sum(v["calibration"] for v in counts.values()),
        "test_total": sum(v["test"] for v in counts.values()),
        "groups": counts,
        "model": args.model,
        "image_size": args.size,
        "preprocess": "letterbox_pad_114+official_normalization",
        "dry_run": args.dry_run,
        "layout": args.layout,
        "seed": args.seed,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("CALIBRATION", sum(v["calibration"] for v in counts.values()),
          "EVALUATION", sum(v["test"] for v in counts.values()),
          "DRY_RUN", args.dry_run)
    print("NOTE: Small hold-out sets are smoke tests; do not claim statistically reliable mAP.")


if __name__ == "__main__":
    main()
