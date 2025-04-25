import os
import shutil
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split ImageNet-val into train/val subsets preserving class folders"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to original ImageNet val directory with class subfolders"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path where split folders will be created"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of images per class to assign to 'train' split"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of moving (preserve original)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Prepare output dirs
    train_root = os.path.join(args.output_dir, 'train')
    val_root   = os.path.join(args.output_dir, 'val')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # Iterate each class
    classes = [d for d in os.listdir(args.input_dir)
               if os.path.isdir(os.path.join(args.input_dir, d))]
    print(f"Found {len(classes)} classes.")

    for cls in classes:
        src_dir = os.path.join(args.input_dir, cls)
        imgs = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpeg')]
        random.shuffle(imgs)

        split_idx = int(len(imgs) * args.train_ratio)
        train_imgs, val_imgs = imgs[:split_idx], imgs[split_idx:]

        # Create class subdirs
        os.makedirs(os.path.join(train_root, cls), exist_ok=True)
        os.makedirs(os.path.join(val_root, cls), exist_ok=True)

        # Move or copy files
        for fname in train_imgs:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(train_root, cls, fname)
            if args.copy:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

        for fname in val_imgs:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(val_root, cls, fname)
            if args.copy:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

        print(f"Class {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

    print("Split completed.")

if __name__ == '__main__':
    main()
