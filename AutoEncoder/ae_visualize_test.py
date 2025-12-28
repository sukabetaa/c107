"""
ae_visualize_test.py

学習済みAutoEncoderを使って、testフォルダの画像を可視化するスクリプト。

- 256x256グレースケールのテスト画像を読み込む
- AutoEncoderで再構成
- 元画像 / 再構成画像 / 誤差ヒートマップ を横に並べた画像を保存
- ファイル名が 'NG' で始まる画像は NG として赤字で表示、それ以外は OK として緑で表示
"""

from pathlib import Path
import os

import cv2
import numpy as np
import torch
import torch.nn as nn


# ==============================
# ★ ここで定数を指定してください
# ==============================
TEST_DIR = r".\images\test"                   # テスト画像が入っているフォルダ
MODEL_PATH = r".\models\cabbage_autoencoder.pt"  # 学習済みAutoEncoder
OUTPUT_DIR = r".\viz_test"                  # 可視化画像の出力先
# ==============================


# ===== AutoEncoder 定義（学習時と同じ構造にする） =====
class CabbageAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [1,256,256]→[16,256,256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # →[16,128,128]

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # →[32,128,128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # →[32,64,64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # →[64,64,64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # →[64,32,32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# →[128,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # →[128,16,16]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # →[64,32,32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # →[32,64,64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # →[16,128,128]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),    # →[1,256,256]
            nn.Sigmoid(),  # [0,1] に収める
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = CabbageAutoEncoder()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def visualize_test_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dir = Path(TEST_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # モデル読み込み
    model = load_model(MODEL_PATH, device)

    # 対象拡張子
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    img_paths = [p for p in sorted(test_dir.iterdir()) if p.suffix.lower() in exts]
    if not img_paths:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(img_paths)} test images.")

    for img_path in img_paths:
        # グレースケールで読み込み（すでに256x256想定）
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"[WARN] Failed to read {img_path}")
            continue

        # [H,W] -> [1,1,H,W], [0,1] に正規化
        img_np = img_gray.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

        # 再構成
        with torch.no_grad():
            recon = model(tensor)

        recon_np = recon.squeeze().cpu().numpy()  # [H,W], 0〜1
        recon_img = np.clip(recon_np * 255.0, 0, 255).astype(np.uint8)

        # ピクセルごとの絶対誤差
        diff = np.abs(img_gray.astype(np.float32) - recon_img.astype(np.float32))
        mse = float((diff ** 2).mean())

        # 誤差ヒートマップ作成（0〜255に正規化してJetカラーマップ）
        if diff.max() > 0:
            diff_norm = (diff / diff.max()) * 255.0
        else:
            diff_norm = diff
        diff_norm = diff_norm.astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

        # 元画像・再構成画像を3chにして並べる
        orig_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        recon_3ch = cv2.cvtColor(recon_img, cv2.COLOR_GRAY2BGR)

        # 横に結合
        viz = np.hstack([orig_3ch, recon_3ch, diff_color])

        # NG/OK 判定（ファイル名の先頭がNGならNG）
        is_ng = img_path.name.startswith("NG")
        label_text = f"{'NG' if is_ng else 'OK'}  mse={mse:.5f}"

        color = (0, 0, 255) if is_ng else (0, 255, 0)  # NG=赤, OK=緑
        cv2.putText(
            viz,
            label_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        # ファイル名に _viz を付けて保存
        out_name = f"{img_path.stem}_viz.png"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), viz)
        print(f"Saved: {out_path}")


def main():
    # models/cabbage_autoencoder.pt と data/test が準備できている前提
    if not os.path.exists(MODEL_PATH):
        print(f"MODEL_PATH '{MODEL_PATH}' not found.")
        return
    if not os.path.exists(TEST_DIR):
        print(f"TEST_DIR '{TEST_DIR}' not found.")
        return

    visualize_test_images()


if __name__ == "__main__":
    main()
