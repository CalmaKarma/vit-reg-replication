import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import multiprocessing

from openclip_with_registers import OpenCLIPWithRegisters

def main():
    # —1— Load OpenCLIP ViT-B/16 with matching preprocess
    backbone, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    model = OpenCLIPWithRegisters(backbone.visual, 0, 1000).cuda().eval()

    # —2— Prepare dataset & loader using the OpenCLIP preprocess transform
    val_dir = "ILSVRC2012/ILSVRC2012_split_0.8/val"
    dataset = ImageFolder(val_dir, transform=preprocess)
    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes.")

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # —3— Collect patch L2 norms
    all_norms = []
    with torch.no_grad():
        for pixels, _ in tqdm(loader, desc="Scanning ImageNet-val", unit="batch"):
            pixels = pixels.cuda(non_blocking=True)
            # get full token features, not the pooled output:
            feats = model.forward_features(pixels)  # (B, 197, D)
            norms = feats[:, 1:, :].norm(dim=-1)  # (B,196)
            all_norms.append(norms.cpu().reshape(-1))

    all_norms = torch.cat(all_norms)
    # —4— Save norms tensor
    torch.save(all_norms, "openclip_all_norms.pt")
    print(f"Saved openclip_all_norms.pt with {all_norms.numel()} tokens.")

def visualize():
    # Load saved norms
    all_norms = torch.load('openclip_all_norms.pt').numpy()

    # Summary statistics
    mean = all_norms.mean()
    std = all_norms.std()
    max_val = all_norms.max()
    p98 = np.percentile(all_norms, 98)

    print(f"Mean = {mean:.2f}")
    print(f"Std  = {std:.2f}")
    print(f"Max  = {max_val:.2f}")
    print(f"98th percentile = {p98:.2f}")

    # Plot histogram: 0 to max+10, show percentage on log scale
    plt.figure(figsize=(8, 5))
    plt.hist(
        all_norms,
        bins=200,
        range=(0, max_val + 10),
        weights=np.ones_like(all_norms) / all_norms.size * 100
    )
    plt.yscale('log')
    plt.ylabel('Percentage of tokens (%) [log scale]')
    plt.xlabel('L2 Norm')
    plt.title('Patch Embedding L2 Norms Distribution (log %)')
    plt.show()

if __name__ == '__main__':
    # multiprocessing.freeze_support()  # on Windows just to be safe
    # main()
    visualize()

