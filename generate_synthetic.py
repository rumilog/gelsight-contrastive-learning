"""
Generate synthetic training data by augmenting phone tiles and GelSight depth
images with MATCHED spatial transforms, preserving cross-modal pairing.

For each (phone_tile, depth_image) pair at the same billet/position:
  - Apply N random spatial transforms identically to both images
  - Apply independent appearance transforms (color, noise) to each
  - Save with naming that preserves the pairing

This turns ~192 phone tiles + ~194 depth images into ~2000+ paired samples
while maintaining the billet/position correspondence needed for cross-modal
contrastive learning.

Usage:
    python generate_synthetic.py                    # defaults (8 augments per pair)
    python generate_synthetic.py --num-augments 12  # more augments
    python generate_synthetic.py --preview          # save a grid preview, don't generate all
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF


REPO_ROOT = Path(__file__).resolve().parent

PHONE_DIRS = {
    "ground":   REPO_ROOT / "camera_images" / "grinded",
    "unground": REPO_ROOT / "camera_images" / "non grinded",
}

GELSIGHT_DIRS = {
    "ground":   REPO_ROOT / "billet_captures_grinded",
    "unground": REPO_ROOT / "billet_captures",
}

OUTPUT_PHONE = {
    "ground":   REPO_ROOT / "synthetic" / "phone" / "grinded",
    "unground": REPO_ROOT / "synthetic" / "phone" / "non grinded",
}

OUTPUT_DEPTH = {
    "ground":   REPO_ROOT / "synthetic" / "depth" / "grinded",
    "unground": REPO_ROOT / "synthetic" / "depth" / "non grinded",
}


def parse_billet_position(filename: str):
    """
    Extract billet number and position from filenames like:
      billet1_camera_3.jpg  -> (1, 3)
      billet1_depth_3.png   -> (1, 3)
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    billet_num = int(parts[0].replace("billet", ""))
    position = int(parts[-1])
    return billet_num, position


def discover_pairs(cls_name: str):
    """
    Find all (phone_tile, depth_image) pairs for a given class
    by matching billet number and position.
    """
    phone_dir = PHONE_DIRS[cls_name]
    depth_dir = GELSIGHT_DIRS[cls_name]

    phone_files = {}
    if phone_dir.exists():
        for f in phone_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                billet, pos = parse_billet_position(f.name)
                phone_files[(billet, pos)] = f

    depth_files = {}
    if depth_dir.exists():
        for f in depth_dir.iterdir():
            if "depth" in f.stem and f.suffix.lower() == ".png":
                billet, pos = parse_billet_position(f.name)
                depth_files[(billet, pos)] = f

    # Match pairs
    pairs = []
    matched_keys = set(phone_files.keys()) & set(depth_files.keys())
    for key in sorted(matched_keys):
        pairs.append({
            "billet": key[0],
            "position": key[1],
            "phone": phone_files[key],
            "depth": depth_files[key],
            "class": cls_name,
        })

    # Unmatched phone tiles (no corresponding depth image)
    phone_only = set(phone_files.keys()) - matched_keys
    # Unmatched depth images (no corresponding phone tile)
    depth_only = set(depth_files.keys()) - matched_keys

    return pairs, phone_only, depth_only


def apply_matched_spatial_transform(phone_img, depth_img, rng):
    """
    Apply identical random spatial transforms to both images.
    Returns (transformed_phone, transformed_depth).
    """
    # Random horizontal flip
    if rng.random() > 0.5:
        phone_img = TF.hflip(phone_img)
        depth_img = TF.hflip(depth_img)

    # Random vertical flip
    if rng.random() > 0.5:
        phone_img = TF.vflip(phone_img)
        depth_img = TF.vflip(depth_img)

    # Random rotation (same angle for both)
    angle = rng.uniform(-30, 30)
    phone_img = TF.rotate(phone_img, angle, fill=0)
    depth_img = TF.rotate(depth_img, angle, fill=0)

    # Random crop — same relative crop for both (accounting for different sizes)
    crop_scale = rng.uniform(0.7, 0.95)

    pw, ph = phone_img.size
    crop_w, crop_h = int(pw * crop_scale), int(ph * crop_scale)
    # Random position as a fraction, applied to both
    frac_x = rng.random()
    frac_y = rng.random()

    p_left = int(frac_x * (pw - crop_w))
    p_top = int(frac_y * (ph - crop_h))
    phone_img = phone_img.crop((p_left, p_top, p_left + crop_w, p_top + crop_h))

    dw, dh = depth_img.size
    d_crop_w, d_crop_h = int(dw * crop_scale), int(dh * crop_scale)
    d_left = int(frac_x * (dw - d_crop_w))
    d_top = int(frac_y * (dh - d_crop_h))
    depth_img = depth_img.crop((d_left, d_top, d_left + d_crop_w, d_top + d_crop_h))

    return phone_img, depth_img


def apply_phone_appearance(img, rng):
    """Apply random appearance transforms to a phone image (independent of depth)."""
    # Brightness
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation
    factor = rng.uniform(0.6, 1.4)
    img = ImageEnhance.Color(img).enhance(factor)

    # Slight blur (simulates focus variation)
    if rng.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    # Add slight noise
    if rng.random() > 0.6:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, rng.uniform(3, 10), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def apply_depth_appearance(img, rng):
    """Apply mild appearance transforms to a depth image."""
    # Slight brightness variation (simulates sensor calibration differences)
    factor = rng.uniform(0.9, 1.1)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Slight contrast
    factor = rng.uniform(0.9, 1.1)
    img = ImageEnhance.Contrast(img).enhance(factor)

    return img


def generate_augmented_pair(pair, aug_idx, rng):
    """Generate one augmented (phone, depth) pair."""
    phone_img = Image.open(pair["phone"]).convert("RGB")
    depth_img = Image.open(pair["depth"]).convert("RGB")

    # Matched spatial transform
    phone_aug, depth_aug = apply_matched_spatial_transform(phone_img, depth_img, rng)

    # Independent appearance transforms
    phone_aug = apply_phone_appearance(phone_aug, rng)
    depth_aug = apply_depth_appearance(depth_aug, rng)

    return phone_aug, depth_aug


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic paired training data")
    parser.add_argument("--num-augments", type=int, default=8,
                        help="Number of augmented versions per original pair")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--preview", action="store_true",
                        help="Only generate a preview grid for one pair")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Discover all cross-modal pairs
    all_pairs = []
    for cls_name in ["ground", "unground"]:
        pairs, phone_only, depth_only = discover_pairs(cls_name)
        all_pairs.extend(pairs)
        print(f"{cls_name}: {len(pairs)} matched pairs, "
              f"{len(phone_only)} phone-only, {len(depth_only)} depth-only")

    print(f"\nTotal matched pairs: {len(all_pairs)}")

    if not all_pairs:
        print("ERROR: No matched pairs found. Check directory structure.")
        return

    if args.preview:
        # Just show one pair with augmentations
        pair = all_pairs[0]
        print(f"\nPreview pair: billet{pair['billet']} pos{pair['position']} ({pair['class']})")
        print(f"  Phone: {pair['phone'].name}")
        print(f"  Depth: {pair['depth'].name}")
        return

    # Create output directories
    for d in list(OUTPUT_PHONE.values()) + list(OUTPUT_DEPTH.values()):
        d.mkdir(parents=True, exist_ok=True)

    # Also copy originals into synthetic dirs (so all training data is in one place)
    total_generated = 0

    for pair in all_pairs:
        cls = pair["class"]
        billet = pair["billet"]
        pos = pair["position"]

        # Copy original pair
        phone_orig = Image.open(pair["phone"]).convert("RGB")
        depth_orig = Image.open(pair["depth"]).convert("RGB")

        phone_name = f"billet{billet}_pos{pos}_orig"
        depth_name = f"billet{billet}_pos{pos}_orig"

        phone_orig.save(OUTPUT_PHONE[cls] / f"{phone_name}.jpg", quality=95)
        depth_orig.save(OUTPUT_DEPTH[cls] / f"{depth_name}.png")

        # Generate augmented versions
        for aug_i in range(args.num_augments):
            phone_aug, depth_aug = generate_augmented_pair(pair, aug_i, rng)

            phone_name = f"billet{billet}_pos{pos}_aug{aug_i+1}"
            depth_name = f"billet{billet}_pos{pos}_aug{aug_i+1}"

            phone_aug.save(OUTPUT_PHONE[cls] / f"{phone_name}.jpg", quality=95)
            depth_aug.save(OUTPUT_DEPTH[cls] / f"{depth_name}.png")

            total_generated += 1

    # Also handle unmatched images (phone tiles without depth, depth without phone)
    for cls_name in ["ground", "unground"]:
        phone_dir = PHONE_DIRS[cls_name]
        depth_dir = GELSIGHT_DIRS[cls_name]
        pairs, phone_only, depth_only = discover_pairs(cls_name)
        matched_keys = {(p["billet"], p["position"]) for p in pairs}

        # Copy unmatched phone tiles
        if phone_dir.exists():
            for f in phone_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    billet, pos = parse_billet_position(f.name)
                    if (billet, pos) not in matched_keys:
                        img = Image.open(f).convert("RGB")
                        img.save(OUTPUT_PHONE[cls_name] / f"billet{billet}_pos{pos}_orig.jpg", quality=95)
                        for aug_i in range(args.num_augments):
                            aug = apply_phone_appearance(img.copy(), rng)
                            aug.save(OUTPUT_PHONE[cls_name] / f"billet{billet}_pos{pos}_aug{aug_i+1}.jpg", quality=95)
                            total_generated += 1

        # Copy unmatched depth images
        if depth_dir.exists():
            for f in depth_dir.iterdir():
                if "depth" in f.stem and f.suffix.lower() == ".png":
                    billet, pos = parse_billet_position(f.name)
                    if (billet, pos) not in matched_keys:
                        img = Image.open(f).convert("RGB")
                        img.save(OUTPUT_DEPTH[cls_name] / f"billet{billet}_pos{pos}_orig.png")
                        for aug_i in range(args.num_augments):
                            aug = apply_depth_appearance(img.copy(), rng)
                            aug.save(OUTPUT_DEPTH[cls_name] / f"billet{billet}_pos{pos}_aug{aug_i+1}.png")
                            total_generated += 1

    # Summary
    print(f"\nGenerated {total_generated} augmented images")
    for label, d in {**OUTPUT_PHONE, **OUTPUT_DEPTH}.items():
        count = len(list(d.iterdir())) if d.exists() else 0
        print(f"  {d.relative_to(REPO_ROOT)}: {count} files")

    print(f"\nSynthetic data saved to: {REPO_ROOT / 'synthetic'}/")
    print("Run train.py to use it for training.")


if __name__ == "__main__":
    main()
