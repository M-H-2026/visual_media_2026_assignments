import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from models.convnext import convnext_tiny


DEFAULT_CHECKPOINT = Path("checkpoints/convnext_tiny_1k_224_ema.pth")
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the authors' ConvNeXt-Tiny on one image using CPU."
    )
    parser.add_argument("image", type=Path, help="Path to an input image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for result files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--using-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the fixed convolution filter before normalization",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = convnext_tiny(num_classes=1000)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=True
    )
    if "model" not in checkpoint:
        raise KeyError("The checkpoint does not contain the expected 'model' key.")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def preprocess_image(image_path: Path) -> torch.Tensor:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = v2.Compose(
        [
            v2.Resize(256, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return transform(image).unsqueeze(0)

def process_image_by_using_filter(image_path: Path) -> torch.Tensor:
    geometric_transform = v2.Compose([
        v2.Resize(256, interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop(224),
        v2.PILToTensor(),
    ])

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = geometric_transform(image)

    # uint8 [3, 224, 224] → float32
    # 値は0～255のまま
    image = image.to(torch.float32)

    # Conv2d用にバッチ次元を追加
    image = image.unsqueeze(0)

    kernel = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0, 9.0, -1.0],
        [-1.0, -1.0,  -1.0],
    ], dtype=torch.float32)

    # [3, 1, 3, 3] にする
    # RGB各チャンネルに同じカーネルを適用
    kernel = kernel.view(1, 1, 3, 3)
    kernel = kernel.repeat(3, 1, 1, 1)

    image = F.conv2d(
        image,
        kernel,
        bias=None,
        stride=1,
        padding=1,
        groups=3,
    )

    # シャープ化で0未満や255超の値が出るため制限
    image = image.clamp(0.0, 255.0)

    # image = F.max_pool2d(
    #     image,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    # )

    # 0～255 → 0～1
    image = image / 255.0

    # ImageNet正規化
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    image = (image - mean) / std

    return image


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    model = load_model(args.checkpoint).to(device)
    if args.using_filter:
        image = process_image_by_using_filter(args.image).to(device)
    else:
        image = preprocess_image(args.image).to(device)


    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1)[0]

    categories = ConvNeXt_Tiny_Weights.DEFAULT.meta["categories"]
    top_probabilities, top_indices = probabilities.topk(5)

    result_lines = [
        f"Image: {args.image.name}",
        f"Device: {device}",
        f"Using filter: {args.using_filter}",
        "Top-5 predictions:",
    ]
    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities.tolist(), top_indices.tolist()), start=1
    ):
        result_lines.append(
            f"{rank}: {categories[class_index]} ({probability * 100:.2f}%)"
        )

    result_text = "\n".join(result_lines) + "\n"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_state = "filter-on" if args.using_filter else "filter-off"
    output_path = (
        args.output_dir / f"{timestamp}_{args.image.stem}_{filter_state}.txt"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_text, encoding="utf-8")

    print(result_text, end="")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
