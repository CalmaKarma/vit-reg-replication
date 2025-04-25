import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import open_clip
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune OpenCLIP ViT-B/16 with register tokens")
    parser.add_argument("--data-dir", type=str, default="ILSVRC2012/ILSVRC2012_split_0.8",
                        help="Root dir of split data (contains 'train' and 'val')")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--num-registers", type=int, default=0,
                        help="Number of register tokens to add")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


class OpenCLIPWithRegisters(nn.Module):
    def __init__(self, vision_model, num_registers, num_classes):
        super().__init__()
        # vision_model is the .visual from open_clip.create_model_and_transforms(...)
        self.vision = vision_model
        hidden = self.vision.transformer.width

        # 1) Registers
        self.registers = nn.Parameter(torch.zeros(1, num_registers, hidden))
        nn.init.trunc_normal_(self.registers, std=0.02)

        # 2) Extend positional embeddings
        old_pos = self.vision.positional_embedding  # (1, 1+n_patches, D) or (seq, D)
        if old_pos.dim() == 3:
            cls_pos = old_pos[:, :1, :]
            patch_pos = old_pos[:, 1:, :]
            reg_pos = torch.zeros(1, num_registers, hidden, device=old_pos.device)
            new_pos = torch.cat([cls_pos, patch_pos, reg_pos], dim=1)
        else:
            cls_pos = old_pos[:1, :]
            patch_pos = old_pos[1:, :]
            reg_pos = torch.zeros(num_registers, hidden, device=old_pos.device)
            new_pos = torch.cat([cls_pos, patch_pos, reg_pos], dim=0)
        self.vision.positional_embedding = nn.Parameter(new_pos)

        # 3) Classifier head
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, pixel_values):
        v = self.vision

        # a) patch conv
        x = v.conv1(pixel_values)  # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, n_patches, D)

        # b) prepend CLS
        ce = v.class_embedding
        if ce.dim() == 1:
            cls = ce.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        elif ce.dim() == 2:
            cls = ce.unsqueeze(0).expand(B, -1, -1)
        else:
            cls = ce.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B,1+n_patches,D)

        # c) append registers
        regs = self.registers.expand(B, -1, -1)  # (B,n_reg,D)
        x = torch.cat([x, regs], dim=1)  # (B,1+n_patches+n_reg,D)

        # d) add pos emb + LN
        x = x + v.positional_embedding
        x = v.ln_pre(x)

        # e) transformer + post-LN
        x = v.transformer(x)
        x = v.ln_post(x[:, 0, :])  # take CLS slot

        # f) classifier
        return self.classifier(x)

    def forward_features(self, pixel_values):
        v = self.vision

        # 1) patch conv → (B, C, H', W')
        x = v.conv1(pixel_values)
        B, C, H, W = x.shape

        # 2) flatten → (B, n_patches, D)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        # 3) prepend CLS
        ce = v.class_embedding
        if ce.dim() == 1:
            cls = ce.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        elif ce.dim() == 2:
            cls = ce.unsqueeze(0).expand(B, -1, -1)
        else:
            cls = ce.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+n_patches, D)

        # 4) append registers
        regs = self.registers.expand(B, -1, -1)  # (B, num_reg, D)
        x = torch.cat([x, regs], dim=1)          # (B, 1+n_patches+R, D)

        # 5) add pos emb + pre-LN
        x = x + v.positional_embedding
        x = v.ln_pre(x)

        # 6) transformer + post-LN
        x = v.transformer(x)
        x = v.ln_post(x)                         # (B, 1+n_patches+R, D)

        return x


def main(num_reg, do_eval=False):
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model_base, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")

    # Data loaders
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    train_ds = ImageFolder(train_dir, transform=preprocess)
    val_ds = ImageFolder(val_dir, transform=preprocess)
    num_classes = len(train_ds.classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True) if do_eval else None

    # Model & optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OpenCLIPWithRegisters(
        vision_model=model_base.visual,  # pass in the ViT only
        num_registers=num_reg,
        num_classes=num_classes
    ).to(device)

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(key in name for key in [
            "registers",
            "classifier",
            "vision.ln_pre",
            "vision.ln_post",
            "vision.positional_embedding",
            "vision.transformer.resblocks.10",
            "vision.transformer.resblocks.11"
        ]):
            param.requires_grad = True

    training_params = []
    untrained_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            training_params.append(name)
        else:
            untrained_params.append(name)
    print('training parameters:', training_params)
    print('untrained parameters', untrained_params)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = total = 0
        for pixels, labels in tqdm(train_loader, desc=f"Train {num_reg} Reg Epoch {epoch + 1}/{args.epochs}"):
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pixels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch + 1} Train Acc: {100 * correct / total:.2f}% Loss: {total_loss / total:.4f}")

        if do_eval:
            model.eval()
            val_loss = correct = total = 0
            with torch.no_grad():
                for pixels, labels in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}"):
                    pixels, labels = pixels.to(device), labels.to(device)
                    logits = model(pixels)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            print(f"Epoch {epoch + 1} Val Acc: {100 * correct / total:.2f}% Loss: {val_loss / total:.4f}\n")

    # Save
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'openclip_{num_reg}_registers.pth'))
    print("Model with registers saved.")


if __name__ == '__main__':
    main(4)
    main(1)
    main(2)

