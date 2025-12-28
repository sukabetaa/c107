import random
import shutil
from pathlib import Path


# ==============================
# ★ ここで定数を指定してください
# ==============================
INPUT_DIR = r"../preprocess/output"   # 元画像フォルダ
OUTPUT_DIR = r"./images"                 # train/val/test の出力先

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

SEED = 42
# ==============================


def split_dataset(input_dir, output_dir,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  seed=42):

    random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must be 1.0"

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]

    if not files:
        raise RuntimeError(f"No image files found in {input_dir}")

    print(f"Found {len(files)} images in {input_dir}")

    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # 誤差吸収

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

    # 出力フォルダ作成
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # コピー処理
    def copy_files(file_list, split_name):
        for src in file_list:
            dst = output_dir / split_name / src.name
            shutil.copy2(src, dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"Dataset split completed and saved to: {output_dir}")


def main():
    split_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
