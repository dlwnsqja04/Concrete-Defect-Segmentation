from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import psutil
import torch
from ultralytics import YOLO, __version__, settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal GPU is unavailable in this terminal session")
    settings.update({"sync": False})
    model = YOLO(str(args.weights.resolve()))
    started = time.monotonic()
    metrics = model.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, optimizer="AdamW", lr0=0.00015, mosaic=0.0, mixup=0.0,
        copy_paste=0.0, close_mosaic=0, patience=20, seed=42, workers=0, cache=False,
        amp=False, project=str(args.output.resolve().parent), name=args.output.name,
        exist_ok=False, plots=False, verbose=False,
    )
    evidence = {
        "seconds": round(time.monotonic() - started, 2),
        "platform": platform.platform(),
        "memory_gb": round(psutil.virtual_memory().total / 1_000_000_000, 2),
        "ultralytics": __version__,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "device": str(model.trainer.device),
        "best": str(model.trainer.best),
        "best_metrics": metrics.results_dict,
    }
    (model.trainer.save_dir / "run_summary.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
