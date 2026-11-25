"""Fine-tuning LR-ASPP CamVid dan inferensi untuk GUI MATLAB."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)


# Nilai RGB asli mengikuti pengelompokan CamVid/SegNet pada contoh MathWorks.
LABEL_GROUPS = {
    "Sky": [(128, 128, 128)],
    "Building": [(128, 0, 0), (0, 128, 64), (64, 192, 0), (64, 0, 64), (192, 0, 128)],
    "Pole": [(192, 192, 128), (0, 0, 64)],
    "Road": [(128, 64, 128), (128, 0, 192), (192, 0, 64)],
    "Pavement": [(0, 0, 192), (64, 192, 128), (128, 128, 192)],
    "Tree": [(128, 128, 0), (192, 192, 0)],
    "SignSymbol": [(192, 128, 128), (128, 128, 64), (0, 64, 64)],
    "Fence": [(64, 64, 128)],
    "Car": [(64, 0, 128), (64, 128, 192), (192, 128, 192), (192, 64, 128), (128, 64, 64)],
    "Pedestrian": [(64, 64, 0), (192, 128, 64), (64, 0, 192), (64, 128, 64)],
    "Bicyclist": [(0, 128, 192), (192, 0, 192)],
}
CLASSES = list(LABEL_GROUPS)
NUM_CLASSES = len(CLASSES)
VOID = 255
ARCHITECTURE = "lraspp_mobilenet_v3_large"
PALETTE = np.zeros((256, 3), dtype=np.uint8)
COLOR_TO_CLASS = np.full((4, 4, 4), VOID, dtype=np.uint8)
for class_id, colors in enumerate(LABEL_GROUPS.values()):
    PALETTE[class_id] = colors[0]
    for color in colors:
        COLOR_TO_CLASS[tuple(channel // 64 for channel in color)] = class_id

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def mask_to_ids(image):
    rgb = np.asarray(image.convert("RGB"))
    r, g, b = np.moveaxis(rgb // 64, -1, 0)
    labels = COLOR_TO_CLASS[r, g, b]
    # Void dan warna yang tidak terdapat pada palet diabaikan dalam loss/IoU.
    labels[np.any(rgb % 64 != 0, axis=-1)] = VOID
    return labels


def image_tensor(image, size):
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255
    return (tensor - MEAN) / STD


def paired_samples(root):
    def index(folder, is_label):
        if not folder.is_dir():
            raise ValueError(f"Folder CamVid tidak ditemukan: {folder}")
        paths = {}
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            name = path.stem.lower()
            if is_label:
                name = name.removesuffix("_l")
            if name in paths:
                raise ValueError(f"Nama frame ganda di {folder}: {name}")
            paths[name] = path
        return paths

    images = index(root / "701_StillsRaw_full", False)
    labels = index(root / "LabeledApproved_full", True)
    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())[:5]
        missing_images = sorted(labels.keys() - images.keys())[:5]
        raise ValueError(f"Pasangan CamVid tidak lengkap. Tanpa label: {missing_labels}. "
                         f"Tanpa citra: {missing_images}.")
    if len(images) < 5:
        raise ValueError("Diperlukan minimal 5 pasangan frame dan label.")
    return {name: (path, labels[name]) for name, path in images.items()}


def partition(names, seed):
    frames = sorted(names)
    random.Random(seed).shuffle(frames)
    n_train = round(0.6 * len(frames))
    n_val = round(0.2 * len(frames))
    return {
        "train": frames[:n_train],
        "validation": frames[n_train:n_train + n_val],
        "test": frames[n_train + n_val:],
    }


class CamVid(Dataset):
    def __init__(self, samples, frame_ids, size, augment=False):
        self.pairs = [samples[name] for name in frame_ids]
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = image_tensor(source, self.size)
        with Image.open(label_path) as source:
            resized = source.resize(self.size, Image.Resampling.NEAREST)
            labels = torch.from_numpy(mask_to_ids(resized)).long()
        if self.augment and random.random() < 0.5:
            image, labels = image.flip(-1), labels.flip(-1)
        return image, labels


def build_model(pretrained=False):
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    if pretrained:
        model = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=None)
        head = model.classifier
        head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, NUM_CLASSES, 1)
        head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, NUM_CLASSES, 1)
        return model
    return lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=NUM_CLASSES)


def update_confusion(confusion, target, pred):
    valid = target != VOID
    codes = NUM_CLASSES * target[valid] + pred[valid]
    counts = torch.bincount(codes, minlength=NUM_CLASSES**2)
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def summarize(confusion):
    true_positive = confusion.diag().numpy()
    union = (confusion.sum(0) + confusion.sum(1) - confusion.diag()).numpy()
    valid = union > 0
    iou = np.zeros(NUM_CLASSES, dtype=float)
    iou[valid] = true_positive[valid] / union[valid]
    pixels = int(confusion.sum())
    return {
        "mean_iou": float(iou[valid].mean()) if valid.any() else None,
        "pixel_accuracy": float(true_positive.sum()) / pixels if pixels else None,
        "iou_per_class": {name: float(iou[i]) if valid[i] else None for i, name in enumerate(CLASSES)},
        "labeled_pixels": pixels,
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def validate(model, batches, device):
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    for inputs, labels in batches:
        inputs, labels = inputs.to(device), labels.to(device)
        update_confusion(confusion, labels, model(inputs)["out"].argmax(1))
    return summarize(confusion)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = paired_samples(args.data)
    splits = partition(samples, args.seed)
    counts = {name: len(frames) for name, frames in splits.items()}
    print(f"CamVid: {counts}, perangkat: {device}")
    size = (args.width, args.height)
    train_set = CamVid(samples, splits["train"], size, augment=True)
    val_set = CamVid(samples, splits["validation"], size)
    train_batches = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers)
    val_batches = DataLoader(val_set, args.batch_size, num_workers=args.workers)
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ])
    criterion = nn.CrossEntropyLoss(ignore_index=VOID)
    history = []
    best_iou = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for inputs, labels in train_batches:
            if not (labels != VOID).any():
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs)["out"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        metrics = validate(model, val_batches, device)
        val_iou = metrics["mean_iou"]
        if steps == 0 or val_iou is None:
            raise ValueError("Mask train atau validation tidak memiliki piksel dari 11 kelas CamVid.")
        record = {"epoch": epoch, "train_loss": total_loss / steps, "validation_mean_iou": val_iou}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs} | loss {record['train_loss']:.4f} | val mIoU {val_iou:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint = {
                "architecture": ARCHITECTURE, "classes": CLASSES, "size": list(size),
                "weights": model.state_dict(), "splits": splits, "seed": args.seed,
                "epoch": epoch, "validation_mean_iou": val_iou,
            }
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(args.checkpoint)
        write_json(args.checkpoint.with_suffix(".training.json"), {"history": history, "splits": splits})
    print(f"Checkpoint dengan val mIoU terbaik: {args.checkpoint}")


def load_model(path, device):
    metadata = torch.load(path, map_location="cpu", weights_only=True)
    if metadata.get("architecture") != ARCHITECTURE or metadata.get("classes") != CLASSES:
        raise ValueError("Checkpoint tidak cocok dengan arsitektur atau urutan kelas CamVid.")
    model = build_model()
    model.load_state_dict(metadata["weights"])
    return model.to(device).eval(), metadata


@torch.inference_mode()
def predict_ids(model, image, size, device):
    inputs = image_tensor(image, size).unsqueeze(0).to(device)
    logits = model(inputs)["out"]
    # Upsampling dilakukan pada logit, sehingga setiap piksel asli memperoleh satu kelas.
    scores = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return scores[0].argmax(0).to(torch.uint8).cpu().numpy()


def predict(args, device):
    model, metadata = load_model(args.checkpoint, device)
    with Image.open(args.image) as source:
        predictions = predict_ids(model, source, metadata["size"], device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(PALETTE[predictions]).save(args.output, format="PNG")
    if args.metadata:
        write_json(args.metadata, {"classes": CLASSES, "colors": PALETTE[:NUM_CLASSES].tolist()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args_by_command = parser.add_subparsers(dest="command", required=True)
    for name, handler in [("train", train), ("predict", predict)]:
        command = args_by_command.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--checkpoint", type=Path, required=True)
        if name == "train":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--epochs", type=int, default=15)
            command.add_argument("--batch-size", type=int, default=8)
            command.add_argument("--width", type=int, default=320)
            command.add_argument("--height", type=int, default=240)
            command.add_argument("--lr", type=float, default=0.001)
            command.add_argument("--seed", type=int, default=4073)
            command.add_argument("--workers", type=int, default=0)
        else:
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--image", type=Path, required=True)
            command.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if args.command == "train" and (min(args.width, args.height) < 32 or args.epochs < 1
                                     or args.batch_size < 1 or args.workers < 0
                                     or not np.isfinite(args.lr) or args.lr <= 0):
        parser.error("Dimensi minimal 32 px, epochs/batch-size positif, workers >= 0, dan lr > 0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA tidak tersedia untuk instalasi PyTorch ini.")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        args.handler(args, device)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Gagal menjalankan CamVid: {error}\n")


if __name__ == "__main__":
    main()
