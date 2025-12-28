import cv2
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# =========================
# 画像読み込み
# =========================
def load_images(input_dir):
    """
    指定フォルダから画像を読み込み、(path, image) のリストを返す
    """
    input_dir = Path(input_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    images = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in exts:
            img = cv2.imread(str(p))
            if img is not None:
                images.append((p, img))
            else:
                logger.debug(f"[WARN] Failed to read {p}")

    logger.debug(f"Loaded {len(images)} images from {input_dir}")
    return images


# =========================
# YOLO推論
# =========================
def run_yolo_inference(model, image, conf=0.25, img_size=640, device="cpu"):
    """
    numpy画像(BGR)をYOLOに渡し、xyxy形式のbbox配列(N,4)を返す
    """
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=img_size,
        device=device,
        verbose=False,
    )

    if not results:
        return np.empty((0, 4))

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4))

    return result.boxes.xyxy.cpu().numpy()


# =========================
# 切り出し＋0パディング
# =========================
def crop_and_pad(image, box, out_size=256):
    h, w = image.shape[:2]

    x1, y1, x2, y2 = box
    x1 = max(int(np.floor(x1)), 0)
    y1 = max(int(np.floor(y1)), 0)
    x2 = min(int(np.ceil(x2)), w)
    y2 = min(int(np.ceil(y2)), h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2, :]
    ch, cw = crop.shape[:2]

    scale = out_size / max(ch, cw)
    new_h = max(int(ch * scale), 1)
    new_w = max(int(cw * scale), 1)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    top = (out_size - new_h) // 2
    left = (out_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = gray

    return canvas


# =========================
# まとめ処理
# =========================
def process_folder(weights, input_dir, output_dir,
                   conf=0.25, img_size=640, out_size=256, device="cpu"):

    model = YOLO(str(weights))

    images = load_images(input_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, img in images:
        logger.debug(f"Processing {img_path.name} ...")

        boxes = run_yolo_inference(model, img, conf=conf,
                                   img_size=img_size, device=device)

        if boxes.shape[0] == 0:
            logger.debug("  No detections.")
            continue

        crop_index = 0
        for box in boxes:
            cropped = crop_and_pad(img, box, out_size=out_size)
            if cropped is None:
                continue

            out_name = f"{img_path.stem}_{crop_index:03d}.png"
            cv2.imwrite(str(output_dir / out_name), cropped)
            crop_index += 1

        logger.debug(f"  Saved {crop_index} crops.")


# =========================
# main: 定数指定
# =========================
def main():
    WEIGHTS   = r".\best.pt" # ファインチューニングしたモデル
    INPUT_DIR = r"..\Cabbage\images" # キャベツの画像が入ったフォルダ
    OUTPUT_DIR = r".\output" # 切り取った画像が保存されるフォルダ

    CONF = 0.25
    IMG_SIZE = 640
    OUT_SIZE = 256
    DEVICE = "cpu"   # GPUなら "0"

    process_folder(
        weights=WEIGHTS,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        conf=CONF,
        img_size=IMG_SIZE,
        out_size=OUT_SIZE,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()