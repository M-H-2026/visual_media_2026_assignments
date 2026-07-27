# ConvNeXt CPU Image Inference

ConvNeXtの著者公式PyTorch実装を利用し、Windows 11のCPU環境で画像1枚をImageNet-1K分類するためのプロジェクトです。

Top-5のクラス名と確率をテキストファイルへ保存します。入力画像へ先鋭化フィルタを適用してから推論を行うことも可能です。

## 元プロジェクト

このプロジェクトは、ConvNeXtの著者公式実装を基にしています。

- 公式リポジトリ: [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- 論文: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- 著者: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie

ConvNeXtのモデル実装と学習済みcheckpointは、元プロジェクトの著者によって作成・公開されたものです。このリポジトリでは、公式の `models/convnext.py` とImageNet-1K学習済み重みを使用しています。

## このリポジトリで追加した機能

- Windows 11のCPU環境で画像1枚を推論
- 著者実装の `convnext_tiny` を直接使用
- ImageNet-1K学習済みcheckpointのCPU読み込み
- Top-5クラス名と確率の表示
- 推論結果のテキストファイル保存
- 実行時刻、入力画像名、フィルタ状態を含む出力ファイル名
- 正規化前に固定画像フィルタを適用するオプション

## 由来と実装・アイデアの内訳

### ConvNeXt原著者による部分

- ConvNeXtのアーキテクチャおよび `models/convnext.py` の公式実装
- ImageNet-1Kで学習されたConvNeXt-Tinyのcheckpoint
- モデルの学習・評価方法

### OpenAI Codex（GPT-5.6 Sol）による部分

- Windows 11のCPU環境で著者実装を動かすためのPython・PyTorch・torchvision・timmの互換性確認と環境整備
- 著者実装の `convnext_tiny` と公式checkpointを用いた、実画像1枚の推論処理の基礎実装
- ImageNetと同じリサイズ、中央切り出し、正規化処理の実装
- ImageNet-1Kのクラス名を用いたTop-5分類結果の表示
- 実行時刻と入力画像名を含むテキストファイルへの結果保存
- `--using-filter` / `--no-using-filter` によるフィルタ処理の切り替えと、出力へのフィルタ状態の記録
- README.mdの生成(一部ユーザーが修正)

### ユーザーによる部分

- 入力画像を0～255の値のまま扱い、ImageNet正規化より前に固定カーネルを適用するという処理方針
- 固定カーネルを `conv2d` の `stride=1`、`padding=1` で適用すれば、画像サイズを維持したままフィルタ処理できるというアイデア
- 上記のアイデアを基にしたフィルタ処理経路の実装
- RGB各チャンネルへ独立して適用する3 × 3先鋭化フィルタを取り入れるというアイデアとカーネルの選定

## 使用モデル

使用するモデルは、ImageNet-1Kで学習されたConvNeXt-Tinyです。

| 項目 | 値 |
|---|---|
| Model | ConvNeXt-Tiny |
| Training dataset | ImageNet-1K |
| Input size | 224 × 224 |
| Number of classes | 1000 |
| Parameters | 約28.6M |
| Execution device | CPU |
| Checkpoint | `convnext_tiny_1k_224_ema.pth` |

ImageNet-22Kで事前学習されたモデルではありません。

## 動作確認環境

- Windows 11
- Python 3.11.5
- PyTorch 2.13.0+cpu
- torchvision 0.28.0+cpu
- timm 0.9.16
- Pillow 12.2.0

## セットアップ

### 1. 仮想環境の作成

PowerShellでリポジトリのルートへ移動し、仮想環境を作成します。

```powershell
cd C:\path\to\ConvNeXt
python -m venv .venv
```

### 2. パッケージのインストール

CPU版PyTorchとtorchvisionをインストールします。

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

続いて、推論に必要なパッケージをインストールします。

```powershell
python -m pip install timm==0.9.16 Pillow
```

依存関係は次のコマンドで確認できます。

```powershell
python -m pip check
```

## Checkpointの準備

著者配布のImageNet-1K学習済みConvNeXt-Tiny checkpointを使用します。

- [convnext_tiny_1k_224_ema.pth](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth)

ダウンロードしたファイルを次の位置へ配置してください。

```text
ConvNeXt/
└── checkpoints/
    └── convnext_tiny_1k_224_ema.pth
```

## 推論方法

### フィルタなし

```powershell
python .\inference.py .\pictures_by_myself\cup.jpg
```

`--no-using-filter` を付けて明示的に無効化することもできます。

```powershell
python .\inference.py .\pictures_by_myself\cup.jpg --no-using-filter
```

### フィルタあり

```powershell
python .\inference.py .\pictures_by_myself\cup.jpg --using-filter
```

現在のフィルタは、RGB各チャンネルへ独立して適用する固定3 × 3畳み込みフィルタです。フィルタ処理はImageNet正規化より前に行われます。

## 推論結果

出力ファイル名には、実行時刻、入力画像名、フィルタ状態が含まれます。

```text
outputs/
├── 20260727_194205_cup_filter-off.txt
└── 20260727_194205_cup_filter-on.txt
```

出力例:

```text
Image: cup.jpg
Device: cpu
Using filter: False
Top-5 predictions:
1: cup (47.96%)
2: measuring cup (21.25%)
3: coffee mug (11.14%)
4: mixing bowl (2.66%)
5: bucket (1.08%)
```

## 画像前処理

フィルタなしの場合は、次の順番で前処理します。

1. 入力画像をRGBへ変換
2. アスペクト比を維持し、短辺を256ピクセルへリサイズ
3. 中央を224 × 224で切り出し
4. `float32` テンソルへ変換し、値を0～1へスケーリング
5. ImageNetの平均値と標準偏差で正規化
6. バッチ次元を追加し、`[1, 3, 224, 224]` としてモデルへ入力

正規化には次の値を使用します。

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

`--using-filter` を指定した場合は、画像を `float32` に変換した後も値を0～255のまま維持し、固定フィルタを適用します。その後、0～1へのスケーリングとImageNet正規化を行います。

## 実装の役割分担

| 処理 | 使用する実装 |
|---|---|
| ConvNeXtモデル構造 | 元プロジェクトの `models/convnext.py` |
| 学習済みパラメーター | 著者配布checkpoint |
| 画像のリサイズ・正規化 | torchvision |
| ImageNet-1Kクラス名 | torchvisionのカテゴリ定義 |
| 単画像推論・結果保存・フィルタ処理 | このリポジトリの `inference.py` |

## License

このプロジェクトは、元のConvNeXtリポジトリのライセンスおよび著作権表示を保持します。詳細は [LICENSE](LICENSE) を参照してください。

## Citation

```bibtex
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022}
}
```
