import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

# 1) pick up the LAION-trained ViT‐B/16 from HF, with attentions on
processor = CLIPProcessor.from_pretrained( "openai/clip-vit-base-patch16" )
attn_model = (
    CLIPVisionModel
    .from_pretrained("openai", output_attentions=True)
    .cuda().eval()
)

# 2) some dataset plumbing
val_dir = "ILSVRC2012/ILSVRC2012_split_0.8/val"
ds      = ImageFolder(val_dir, transform=lambda img: processor(images=img, return_tensors="pt")["pixel_values"][0])
loader  = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# 3) scan a subset, record (idx, max_norm)
max_norms = []
with torch.no_grad():
    for idx, (pixels, _) in enumerate(tqdm(loader, total=1000, desc="Scanning subset")):
        if idx >= 1000: break
        pix = pixels.cuda()
        out = attn_model(pix)
        # out.last_hidden_state is (1,197,D) so
        norms = out.last_hidden_state[:,1:,:].norm(dim=-1)        # (1,196)
        max_norms.append((idx, float(norms.max())))
# sort
max_norms = sorted(max_norms, key=lambda x: x[1], reverse=True)

# 4) plot top_k attention overlays
top_k = 4
fig, axes = plt.subplots(top_k, 3, figsize=(15,5*top_k))
for row, (idx, norm_val) in enumerate(max_norms[:top_k]):
    # pull the raw PIL image out of the dataset
    img, _ = ds.dataset.imgs[idx]      # ds.dataset is ImageFolder
    img = Image.open(img).convert("RGB")

    # re-run through HF model to get attentions
    inputs = processor(images=img, return_tensors="pt")["pixel_values"].cuda()
    with torch.no_grad():
        out   = attn_model(inputs)
    attns = out.attentions[-1][0]       # (heads, 197,197)
    cls_attn = attns[:,0,1:].mean(0).cpu().numpy()  # (196,)
    cls_attn = cls_attn.reshape(14,14)

    # upsample
    hm = cls_attn.astype(np.float32)
    heatmap = Image.fromarray(hm, mode="F").resize(img.size, Image.BICUBIC)
    heat_arr = np.array(heatmap)
    heat_arr = (heat_arr - heat_arr.min())/(heat_arr.max()-heat_arr.min()+1e-6)

    # col 0: raw image
    axes[row,0].imshow(img); axes[row,0].axis("off")
    axes[row,0].set_title(f"Idx={idx}, max norm={norm_val:.1f}")

    # col 1: overlay
    axes[row,1].imshow(img)
    axes[row,1].imshow(heat_arr, cmap="jet", alpha=0.4)
    axes[row,1].axis("off")
    axes[row,1].set_title("Last-layer [CLS] Attn")

    # col 2: raw heatmap
    im = axes[row,2].imshow(cls_attn, cmap="viridis", interpolation="nearest")
    axes[row,2].axis("off")
    axes[row,2].set_title("Raw [CLS] Attn")
    fig.colorbar(im, ax=axes[row,2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
