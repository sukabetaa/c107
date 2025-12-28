from ultralytics import YOLO
from pathlib import Path


def main():
    # 1) 事前学習済みモデルを読み込み
    #    軽量なら yolov8n.pt / もう少し精度なら yolov8s.pt など
    model = YOLO("yolov8n.pt")  # or "yolov8s.pt"

    # 2) データ設定ファイル（先ほど作った cabbage.yaml）
    data_yaml = Path("cabbage.yaml")  # スクリプトと同じフォルダに置いた想定

    # 3) 学習設定
    results = model.train(
        data=str(data_yaml),    # データセット設定
        epochs=50,              # エポック数（PCに合わせて調整）
        imgsz=640,              # 入力画像サイズ
        batch=16,               # バッチサイズ（VRAMに合わせて調整）
        project="runs_cabbage", # 結果保存用フォルダ名
        name="yolov8n_cabbage", # 実験名
        patience=20,            # 早期終了の我慢エポック数
        lr0=0.01,               # 初期学習率(デフォルトのままでもOK)
        weight_decay=0.0005,    # L2正則化
        device=0,               # GPU:0 を使用（CPUなら 'cpu'）
    )

    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()
