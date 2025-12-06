#!/usr/bin/env python3
import argparse, shutil, pathlib, random, sys

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def find_image_for_label(label_path: pathlib.Path, images_dir: pathlib.Path):
    stem = label_path.stem
    # If labels have a prefix like 'abcd-IMG_001', drop it
    stem_candidate = stem.split('-', 1)[1] if '-' in stem else stem

    # exact stem match first
    candidates = []
    for img in images_dir.rglob("*"):
        if img.suffix.lower() in IMG_EXTS and (img.stem == stem or img.stem == stem_candidate):
            candidates.append(img)
    if candidates:
        candidates.sort(key=lambda p: (len(str(p)), p.name))
        return candidates[0]

    # fallback: substring match
    for img in images_dir.rglob("*"):
        if img.suffix.lower() in IMG_EXTS and stem_candidate in img.stem:
            return img
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=pathlib.Path, required=True)
    ap.add_argument('--labels', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--val_split', type=float, default=0.2)
    a = ap.parse_args()

    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (a.out / p).mkdir(parents=True, exist_ok=True)

    matched, missing = [], []
    for lf in sorted(a.labels.glob("*.txt")):
        img = find_image_for_label(lf, a.images)
        (matched if img else missing).append((img, lf) if img else lf.name)

    if not matched:
        print("No pairs matched. Check that image names match label stems.")
        sys.exit(1)

    random.seed(0xC0FFEE)
    random.shuffle(matched)
    n_val = max(1, int(len(matched) * a.val_split))
    val_pairs, train_pairs = matched[:n_val], matched[n_val:]

    def cp(pair, split):
        img, lbl = pair
        shutil.copy2(img, a.out / f"images/{split}/{img.stem}{img.suffix.lower()}")
        shutil.copy2(lbl, a.out / f"labels/{split}/{img.stem}.txt")

    for pr in train_pairs: cp(pr, "train")
    for pr in val_pairs:   cp(pr, "val")

    print(f"Prepared at: {a.out}")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    if missing:
        print("\nLabels with no matching image:")
        for m in missing[:20]: print(" -", m)

if __name__ == "__main__":
    main()
