"""
Run inference on a phone camera image to predict whether a billet surface
is ground (smooth) or needs grinding (unground/rough).

Uses the backbone + classifier from the 2-stage contrastive training pipeline.

Usage:
    python predict.py path/to/photo.jpg
    python predict.py path/to/photo.jpg --model trained_models/best_model.pt
    python predict.py path/to/folder/       # batch predict all images in folder
    python predict.py photo.jpg --tta       # test-time augmentation for robustness
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

IMG_SIZE = 224


class Classifier(nn.Module):
    """
    Same classifier head architecture used in train.py:
      head = Dropout(0.3) -> Linear(feature_dim, 2)
    """

    def __init__(self, in_dim, num_classes: int = 2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained backbone + classifier from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild backbone (EfficientNet-B0 with Identity classifier)
    backbone = models.efficientnet_b0(weights=None)
    feature_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.to(device)
    backbone.eval()

    # Rebuild classifier head with same structure as during training
    classifier = Classifier(feature_dim, num_classes=2)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.to(device)
    classifier.eval()

    class_names = checkpoint.get("class_names", ["ground (smooth)", "unground (rough)"])
    return backbone, classifier, class_names


def get_inference_transform():
    """Standard inference preprocessing."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_single(backbone, classifier, image_path: str, transform, device, class_names):
    """Predict a single image and return results."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    features = backbone(input_tensor)
    logits = classifier(features)
    probs = F.softmax(logits, dim=1)[0]

    predicted_idx = probs.argmax().item()
    confidence = probs[predicted_idx].item()

    return {
        "file": str(image_path),
        "prediction": class_names[predicted_idx],
        "confidence": confidence,
        "needs_grinding": predicted_idx == 1,
        "probabilities": {name: round(probs[i].item(), 4) for i, name in enumerate(class_names)},
    }


@torch.no_grad()
def predict_with_tta(backbone, classifier, image_path: str, device, class_names, n_augments=10):
    """
    Test-Time Augmentation: run multiple augmented versions of the same image
    and average predictions for more robust results.
    """
    image = Image.open(image_path).convert("RGB")

    tta_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    center_transform = get_inference_transform()
    all_probs = []

    # Standard center crop
    input_tensor = center_transform(image).unsqueeze(0).to(device)
    features = backbone(input_tensor)
    logits = classifier(features)
    all_probs.append(F.softmax(logits, dim=1)[0])

    # Augmented versions
    for _ in range(n_augments - 1):
        input_tensor = tta_transform(image).unsqueeze(0).to(device)
        features = backbone(input_tensor)
        logits = classifier(features)
        all_probs.append(F.softmax(logits, dim=1)[0])

    avg_probs = torch.stack(all_probs).mean(dim=0)
    predicted_idx = avg_probs.argmax().item()
    confidence = avg_probs[predicted_idx].item()

    return {
        "file": str(image_path),
        "prediction": class_names[predicted_idx],
        "confidence": confidence,
        "needs_grinding": predicted_idx == 1,
        "probabilities": {name: round(avg_probs[i].item(), 4) for i, name in enumerate(class_names)},
        "tta_samples": n_augments,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict billet surface condition")
    parser.add_argument("input", type=str, help="Path to an image or folder of images")
    parser.add_argument("--model", type=str, default="trained_models/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--tta", action="store_true",
                        help="Use Test-Time Augmentation for more robust predictions")
    parser.add_argument("--tta-samples", type=int, default=10,
                        help="Number of augmented samples for TTA")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Run train.py first to create a trained model.")
        sys.exit(1)

    backbone, classifier, class_names = load_model(args.model, device)
    transform = get_inference_transform()

    input_path = Path(args.input)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        )
    else:
        print(f"ERROR: {args.input} is not a valid file or directory")
        sys.exit(1)

    if not image_paths:
        print("No images found.")
        sys.exit(1)

    print(f"\nModel: {args.model}")
    print(f"Device: {device}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    print(f"{'='*60}\n")

    for img_path in image_paths:
        if args.tta:
            result = predict_with_tta(backbone, classifier, str(img_path), device, class_names, args.tta_samples)
        else:
            result = predict_single(backbone, classifier, str(img_path), transform, device, class_names)

        status = "NEEDS GRINDING" if result["needs_grinding"] else "OK (ground)"
        print(f"  {img_path.name}")
        print(f"    Prediction:  {status}")
        print(f"    Confidence:  {result['confidence']:.1%}")
        print(f"    Probs:       {result['probabilities']}")
        print()


if __name__ == "__main__":
    main()
