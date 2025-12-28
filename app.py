import csv
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from AutoEncoder.train_ae import CabbageAutoEncoder
from preprocess.crop_image import crop_and_pad, run_yolo_inference

# ============ Settings ============
WATCH_DIR = Path("./watch_input")
OUTPUT_DIR = Path("./watch_output")
EXT = ".mp4"

YOLO_WEIGHTS = Path("./preprocess/best.pt")
AUTOENCODER_WEIGHTS = Path("./AutoEncoder/models/cabbage_autoencoder.pt")
ANOMALY_SCORES = Path("./AutoEncoder/anomaly_scores.csv")

YOLO_CONF = 0.25
YOLO_IMG_SIZE = 640
AE_CROP_SIZE = 256
ANOMALY_QUANTILE = 0.95
ANOMALY_FALLBACK = 0.0175  # fallback MSE threshold when CSV is missing
# ==================================

_yolo_model: Optional[YOLO] = None
_ae_model: Optional[CabbageAutoEncoder] = None
_device: Optional[torch.device] = None
_anomaly_threshold: Optional[float] = None


def _get_device() -> torch.device:
    """Pick GPU if available, otherwise CPU."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {_device}")
    return _device


def _load_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_WEIGHTS.exists():
            raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
        print(f"[INFO] YOLO model loaded from {YOLO_WEIGHTS}")
    return _yolo_model


def _load_autoencoder(device: torch.device) -> CabbageAutoEncoder:
    global _ae_model
    if _ae_model is None:
        if not AUTOENCODER_WEIGHTS.exists():
            raise FileNotFoundError(f"AutoEncoder weights not found: {AUTOENCODER_WEIGHTS}")
        model = CabbageAutoEncoder()
        state = torch.load(AUTOENCODER_WEIGHTS, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _ae_model = model
        print(f"[INFO] AutoEncoder loaded from {AUTOENCODER_WEIGHTS}")
    return _ae_model


def _load_anomaly_threshold() -> float:
    global _anomaly_threshold
    if _anomaly_threshold is not None:
        return _anomaly_threshold

    if ANOMALY_SCORES.exists():
        try:
            with ANOMALY_SCORES.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                scores = [float(row["mse"]) for row in reader if "mse" in row]
            if scores:
                _anomaly_threshold = float(np.quantile(scores, ANOMALY_QUANTILE))
                print(
                    f"[INFO] Loaded anomaly threshold from CSV "
                    f"(q={ANOMALY_QUANTILE:.2f}): {_anomaly_threshold:.5f}"
                )
                return _anomaly_threshold
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARN] Failed to read anomaly scores CSV: {e}")

    _anomaly_threshold = ANOMALY_FALLBACK
    print(f"[INFO] Using fallback anomaly threshold: {_anomaly_threshold:.5f}")
    return _anomaly_threshold


def _score_crop(crop_gray: np.ndarray, device: torch.device) -> float:
    """Run the AutoEncoder and return reconstruction MSE for a single crop."""
    model = _load_autoencoder(device)
    tensor = torch.from_numpy(crop_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2).item()
    return mse


def _annotate_anomalies(frame: np.ndarray, anomalies: Iterable[Tuple[np.ndarray, float]]) -> np.ndarray:
    annotated = frame.copy()
    for box, mse in anomalies:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"NG {mse:.4f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def process_video(video_path: Path):
    """
    Detect cabbages with YOLO, crop them, score with AutoEncoder, and export annotated video.
    """
    device = _get_device()
    _load_yolo_model()
    _load_autoencoder(device)
    threshold = _load_anomaly_threshold()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / video_path.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    yolo_device = "0" if device.type == "cuda" else "cpu"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = run_yolo_inference(
            model=_load_yolo_model(),
            image=frame,
            conf=YOLO_CONF,
            img_size=YOLO_IMG_SIZE,
            device=yolo_device,
        )

        anomalies = []
        for box in boxes:
            crop = crop_and_pad(frame, box, out_size=AE_CROP_SIZE)
            if crop is None:
                continue
            mse = _score_crop(crop, device)
            if mse >= threshold:
                anomalies.append((box, mse))

        annotated = _annotate_anomalies(frame, anomalies) if anomalies else frame
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print(
        f"[DONE] Processed {frame_idx} frames from {video_path.name}. "
        f"Annotated video saved to {out_path} (threshold={threshold:.5f})."
    )


class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Handle newly created files in the watch folder."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() != EXT:
            return

        print(f"[DETECTED] New file: {path.name}")
        self._wait_until_stable(path)
        process_video(path)

    def _wait_until_stable(self, path: Path, wait=0.5, retry=10):
        """Wait until file size stops growing before processing."""
        prev_size = -1
        for _ in range(retry):
            size = path.stat().st_size
            if size == prev_size:
                return
            prev_size = size
            time.sleep(wait)


def main():
    WATCH_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"[START] Watching folder: {WATCH_DIR.resolve()}")

    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
