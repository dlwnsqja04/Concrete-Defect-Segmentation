from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Final

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import ultralytics
from PIL import Image, ImageDraw
from ultralytics import YOLO

from code_metrics import counts, select_threshold
from prepare_data import CLASSES

GRID: Final = (0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CONF_FLOOR: Final = 0.0009
NMS_IOU: Final = 0.7
MAX_DET: Final = 300
IMGSZ: Final = 640


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, ...]
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class Frame:
    path: Path
    polygons: dict[int, list[tuple[tuple[float, float], ...]]]
    detections: tuple[Detection, ...]

    def score(self, class_id: int) -> float:
        return max((d.confidence for d in self.detections if d.class_id == class_id), default=0.0)


def read_frames(model: YOLO, root: Path, split: str, device: str) -> list[Frame]:
    """Infer a split once and keep detections for threshold scoring and figures."""
    paths = sorted((root / "images" / split).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(root / "images" / split)
    predictions = model.predict(
        [str(path) for path in paths], stream=True, batch=4, imgsz=IMGSZ,
        device=device, conf=CONF_FLOOR, iou=NMS_IOU, max_det=MAX_DET, verbose=False,
    )
    frames = []
    for path, result in zip(paths, predictions, strict=True):
        if path.resolve() != Path(result.path).resolve():
            raise ValueError(f"Unexpected prediction order: {result.path}")
        polygons = {}
        label = root / "labels" / split / path.with_suffix(".txt").name
        for line in label.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            cid, coords = int(tokens[0]), tuple(map(float, tokens[1:]))
            polygons.setdefault(cid, []).append(tuple(zip(coords[::2], coords[1::2], strict=True)))

        boxes = result.boxes.cpu()
        contours = result.masks.xy if result.masks is not None else [()] * len(boxes)
        detections = tuple(
            Detection(int(cid), float(score), tuple(box), tuple((float(x), float(y)) for x, y in contour))
            for cid, score, box, contour in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), contours, strict=True
            )
        )
        frames.append(Frame(path, polygons, detections))
    return frames


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_scores(output: Path, split: str, frames: list[Frame]) -> None:
    write_csv(output / f"{split}_scores.csv", ["class", "image", "actual_on", "max_confidence"], [
        [name, frame.path.name, int(cid in frame.polygons), frame.score(cid)]
        for cid, name in enumerate(CLASSES) for frame in frames
    ])


def split_fingerprint(root: Path, split: str) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "images" / split).glob("*.jpg")):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(hashlib.sha256((root / "labels" / split / path.with_suffix(".txt").name).read_bytes()).digest())
    return digest.hexdigest()


def tune(model: YOLO, root: Path, output: Path, device: str, config: dict) -> None:
    frames = read_frames(model, root, "val", device)
    write_scores(output, "val", frames)
    chosen = {}
    rows = []
    fig, ax = plt.subplots(figsize=(8.2, 3.7), layout="constrained")
    for cid, name in enumerate(CLASSES):
        truth = [cid in frame.polygons for frame in frames]
        scores = [frame.score(cid) for frame in frames]
        threshold = select_threshold(truth, scores, GRID)
        chosen[name] = {"threshold": threshold, **counts(truth, scores, threshold)}
        curve = [counts(truth, scores, t) for t in GRID]
        rows.extend([name, t, *row.values()] for t, row in zip(GRID, curve, strict=True))
        line, = ax.plot(GRID, [row["f1"] for row in curve], marker="o", markersize=3, label=name)
        ax.scatter([threshold], [chosen[name]["f1"]], s=85, facecolors="white",
                   edgecolors=line.get_color(), linewidths=2, zorder=4)
    ax.set(xscale="log", xlabel="Confidence threshold (log scale)", ylabel="Image-level F1", ylim=(-0.03, 1.04))
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="lower left", ncol=3, frameon=True, facecolor="white")
    fig.savefig(output / "val_threshold_f1.png", dpi=210)
    plt.close(fig)
    write_csv(output / "val_threshold_f1.csv", ["class", "threshold", *list(counts([], [], 0))], rows)
    locked = {"selection_split": "val", "tie_break": "lower threshold", "grid": GRID,
              "validation_images": len(frames), "config": config, "classes": chosen}
    with (output / "thresholds.json").open("x", encoding="utf-8") as file:
        json.dump(locked, file, indent=2)
    print(json.dumps(locked, indent=2))


def render_error(frame: Frame, cid: int, kind: str, output: Path) -> None:
    """Overlay the target-class prediction for FP or the target annotation for FN."""
    with Image.open(frame.path) as original:
        image = original.convert("RGBA")
    layer = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    if kind == "FP":
        detection = max((d for d in frame.detections if d.class_id == cid), key=lambda d: d.confidence)
        if detection.polygon:
            draw.polygon(detection.polygon, fill=(241, 80, 71, 85), outline=(255, 131, 115, 255), width=5)
        draw.rectangle(detection.box, outline=(255, 131, 115, 255), width=4)
        x0, y0, x1, y1 = detection.box
    else:
        vertices = []
        for polygon in frame.polygons[cid]:
            points = [(x * width, y * height) for x, y in polygon]
            vertices.extend(points)
            draw.polygon(points, fill=(237, 204, 48, 90), outline=(255, 232, 87, 255), width=5)
        xs, ys = zip(*vertices)
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    pad_x, pad_y = max(55, (x1 - x0) * 0.30), max(55, (y1 - y0) * 0.30)
    crop = (max(0, round(x0 - pad_x)), max(0, round(y0 - pad_y)),
            min(width, round(x1 + pad_x)), min(height, round(y1 + pad_y)))
    combined = Image.alpha_composite(image, layer).convert("RGB")
    combined = combined.crop(crop)
    combined.thumbnail((720, 520), Image.Resampling.LANCZOS)
    combined.save(output, quality=93)


def evaluate(model: YOLO, root: Path, output: Path, device: str, config: dict) -> None:
    threshold_path = output / "thresholds.json"
    locked = json.loads(threshold_path.read_text(encoding="utf-8"))
    if locked["selection_split"] != "val" or locked["config"] != config:
        raise ValueError("Threshold file does not match the validation split and inference settings")
    test_thresholds = {name: entry["threshold"] for name, entry in locked["classes"].items()}
    frames = read_frames(model, root, "test", device)
    write_scores(output, "test", frames)
    rows = []
    errors = []
    (output / "error_cases").mkdir()
    for cid, name in enumerate(CLASSES):
        threshold = test_thresholds[name]
        truth = [cid in frame.polygons for frame in frames]
        scores = [frame.score(cid) for frame in frames]
        record = counts(truth, scores, threshold)
        rows.append([name, threshold, *record.values()])
        for kind, selected in (
            ("FP", sorted((f for f in frames if cid not in f.polygons and f.score(cid) >= threshold),
                          key=lambda f: (-f.score(cid), f.path.name))),
            ("FN", sorted((f for f in frames if cid in f.polygons and f.score(cid) < threshold),
                          key=lambda f: (f.score(cid), f.path.name))),
        ):
            for position, frame in enumerate(selected[:3], 1):
                filename = f"{name}_{kind}{position}.jpg"
                render_error(frame, cid, kind, output / "error_cases" / filename)
                errors.append([name, kind, frame.path.name, frame.score(cid), threshold, filename])
    write_csv(output / "test_onoff.csv", ["class", "threshold", *list(counts([], [], 0))], rows)
    write_csv(output / "error_cases.csv", ["class", "error", "image", "score", "threshold", "file"], errors)
    metrics = model.val(data=str(root / "data.yaml"), split="test", imgsz=IMGSZ, device=device,
                        batch=4, workers=0, conf=0.001, iou=NMS_IOU, max_det=MAX_DET,
                        plots=False, project=str(output), name="mask_evaluation", verbose=False)
    seg = metrics.seg
    by_class = {
        CLASSES[int(cid)]: {"mask_AP50": float(ap50), "mask_AP50_95": float(ap)}
        for cid, ap50, ap in zip(seg.ap_class_index, seg.ap50, seg.ap, strict=True)
    }
    summary = {"test_images": len(frames), "mask_mAP50": float(seg.map50), "mask_mAP50_95": float(seg.map),
               "classes": by_class, "macro_f1": sum(row[-1] for row in rows) / len(rows),
               "thresholds_sha256": hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
               "test_sha256": split_fingerprint(root, "test"),
               "platform": platform.platform(), "python": platform.python_version(),
               "torch": str(torch.__version__), "ultralytics": ultralytics.__version__, "device": device}
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune on validation images, then evaluate with locked thresholds.")
    parser.add_argument("stage", choices=("tune", "test"))
    parser.add_argument("--data", required=True, type=Path, help="Prepared dataset directory")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("This session cannot access MPS. Use an MPS-enabled terminal or --device cpu.")
    ultralytics.settings.update({"sync": False})
    model = YOLO(str(args.weights.resolve(strict=True)))
    if model.task != "segment" or model.names != dict(enumerate(CLASSES)):
        raise ValueError("Expected the 3-class Week 2 segmentation checkpoint")
    config = {"weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "val_sha256": split_fingerprint(args.data, "val"), "imgsz": IMGSZ, "device": args.device,
              "conf_floor": CONF_FLOOR, "nms_iou": NMS_IOU, "max_det": MAX_DET}
    output = args.output.resolve()
    if args.stage == "tune":
        output.mkdir(parents=True, exist_ok=False)
        tune(model, args.data.resolve(), output, args.device, config)
    else:
        evaluate(model, args.data.resolve(), output, args.device, config)


if __name__ == "__main__":
    main()
