"""
cabbage_autoencoder.py

256x256 グレースケールのキャベツ画像を使って
PyTorch の AutoEncoder で「異常検知」を行うための学習＆推論スクリプトです。

前提：
- すでに YOLO で切り出し＆前処理済みの 256x256 グレースケール画像がある
- 学習には「正常なキャベツ」だけを使う（異常なしデータ）
- 異常検知は「再構成誤差（MSE）」をスコアとして使う

ディレクトリ構成（例）：
  data/
    train/   … 正常キャベツ画像だけ（学習用）
    val/     … 正常キャベツ画像（検証用、あれば）
    test/    … 正常＋異常の混在（再構成誤差を CSV に出力）

※ 必要に応じてパスやパラメータは main() の定数を書き換えてください。
"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv


# =========================
# Dataset
# =========================

class CabbageDataset(Dataset):
    """
    256x256 グレースケール画像を読み込む Dataset

    - 全ての画像は 1ch (グレースケール) Tensor [1, 256, 256] に変換
    - 正規化は [0, 1] にスケーリングのみ（AutoEncoder なので十分）
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_paths: List[Path] = [
            p for p in sorted(self.root_dir.glob("*")) if p.suffix.lower() in exts
        ]
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.root_dir}")

        # 256x256 グレースケール + テンソル化
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 安全のためもう一度グレースケール化
            transforms.ToTensor(),                        # [0,1] の float32, shape: [1, H, W]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img  # AutoEncoder なので label は不要


# =========================
# AutoEncoder モデル
# =========================

class CabbageAutoEncoder(nn.Module):
    """
    1x256x256 のグレースケール画像用のシンプルな Conv AutoEncoder

    Encoder: 256x256 → 128 → 64 → 32 → 16
    Decoder: 16 → 32 → 64 → 128 → 256 に復元
    """

    def __init__(self):
        super().__init__()

        # Encoder: Conv + ReLU + MaxPool
        self.encoder = nn.Sequential(
            # [1, 256, 256] -> [16, 128, 128]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # [16, 128, 128] -> [32, 64, 64]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # [32, 64, 64] -> [64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # [64, 32, 32] -> [128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Decoder: ConvTranspose + ReLU
        self.decoder = nn.Sequential(
            # [128, 16, 16] -> [64, 32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),

            # [64, 32, 32] -> [32, 64, 64]
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),

            # [32, 64, 64] -> [16, 128, 128]
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            # [16, 128, 128] -> [1, 256, 256]
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),  # 出力を [0,1] に収める
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# =========================
# 学習ループ
# =========================

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    save_path: str = "cabbage_autoencoder.pt",
):
    """
    AutoEncoder の学習を行う
    - 損失関数は MSELoss（再構成誤差）
    - ベストな validation loss のモデルを保存
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ------- Train -------
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ------- Validation -------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_loss += loss.item() * batch.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # ベストモデルを保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> best model updated (val_loss={val_loss:.6f}) saved to {save_path}")

    print("Training finished.")
    print(f"Best val_loss = {best_val_loss:.6f}")


# =========================
# 異常検知（再構成誤差の計算）
# =========================

def compute_reconstruction_errors(
    model: nn.Module,
    data_dir: str,
    device: torch.device,
    csv_path: str = "anomaly_scores.csv",
):
    """
    data_dir 内の全画像について、
    - AutoEncoder の出力との MSE を計算
    - ファイル名とスコアを CSV で保存

    CSV フォーマット:
        filename, mse
    """
    dataset = CabbageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Dataset 内部の image_paths を利用してファイル名を取得する
    image_paths: List[Path] = dataset.image_paths

    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction="mean")

    records: List[Tuple[str, float]] = []

    with torch.no_grad():
        for (img_tensor,), img_path in zip(loader, image_paths):
            img_tensor = img_tensor.to(device)
            recon = model(img_tensor)
            loss = criterion(recon, img_tensor).item()
            records.append((img_path.name, loss))

    # CSV に書き出し
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "mse"])
        for filename, mse in records:
            writer.writerow([filename, mse])

    print(f"Reconstruction errors have been saved to {csv_path}")
    print("Top 5 highest errors (likely anomalies):")
    for filename, mse in sorted(records, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {filename}: mse={mse:.6f}")


# =========================
# main: 定数をここで指定
# =========================

def main():
    # ---- データパス（必要に応じて書き換え） ----
    TRAIN_DIR = r".\images\train"   # 正常キャベツ（学習用）
    VAL_DIR   = r".\images\val"     # 正常キャベツ（検証用）
    TEST_DIR  = r".\images\test"    # 正常＋異常混在（スコア出力用）

    # ---- モデル保存先 ----
    MODEL_PATH = r".\models\cabbage_autoencoder.pt"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # ---- 学習パラメータ ----
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3

    # ---- デバイス ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Dataset / DataLoader 作成 ----
    train_dataset = CabbageDataset(TRAIN_DIR)
    val_dataset = CabbageDataset(VAL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # ---- モデル定義＆学習 ----
    model = CabbageAutoEncoder()
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        save_path=MODEL_PATH,
    )

    # ---- 異常検知用のスコア計算 ----
    if Path(TEST_DIR).exists():
        print("Computing reconstruction errors on TEST_DIR for anomaly detection...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        compute_reconstruction_errors(
            model=model,
            data_dir=TEST_DIR,
            device=device,
            csv_path="anomaly_scores.csv",
        )
    else:
        print(f"TEST_DIR '{TEST_DIR}' not found. Skipping anomaly score evaluation.")


if __name__ == "__main__":
    main()
