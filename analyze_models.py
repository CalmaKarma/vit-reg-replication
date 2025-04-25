import os, warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import timm

# 1) TensorFlow oneDNN & deprecation logs
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '2'   # 0=all, 1=info, 2=warning, 3=error

# 2) Python warnings filter
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

val_dir = "ILSVRC2012/ILSVRC2012_split_0.8/val"
transform = Compose([
    Resize(256, interpolation=3),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=(0.485,0.456,0.406),
              std =(0.229,0.224,0.225)),
])
val_ds = ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                        num_workers=0, pin_memory=True)

dinov2_base = torch.hub.load('facebookresearch/dinov2','dinov2_vitb14',pretrained=False).cuda().eval()
state_base = torch.load("dinov2_vitb14_pretrain.pth", map_location="cpu")
dinov2_base.load_state_dict(state_base, strict=False)
dinov2_base.eval().cuda()

dinov2_reg4   = torch.hub.load('facebookresearch/dinov2','dinov2_vitb14_reg',pretrained=False).cuda().eval()
state_reg4 = torch.load("dinov2_vitb14_reg4_pretrain.pth", map_location="cpu")
dinov2_reg4.load_state_dict(state_reg4, strict=False)
dinov2_reg4.eval().cuda()

def compute_patch_norms(model):
    all_norms = []
    with torch.no_grad():
        for imgs, _ in tqdm(val_loader, desc="Norms", unit="batch"):
            imgs = imgs.cuda(non_blocking=True)
            out = model.forward_features(imgs)
            # print(type(out), out)
            # print(out.items())
            patch_feats = out['x_norm_patchtokens']         # drop CLS and any extra registers
            norms = patch_feats.norm(dim=-1).cpu().view(-1)
            all_norms.append(norms)
    return np.concatenate(all_norms, axis=0)

norms_base = compute_patch_norms(dinov2_base)
norms_reg4 = compute_patch_norms(dinov2_reg4)