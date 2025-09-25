"""
train_hrnet_pluck.py

Train HRNet-W18 backbone (from timm) + regression head to predict plucking point (x,y)
normalized in crop coordinates.

Usage:
    python train_hrnet_pluck.py --data_dir final_dataset --epochs 50 --batch_size 16 --img_size 256 --out model_pluck.pth
"""

import os
import argparse
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import timm
from timm.data import resolve_data_config, create_transform

# ---------------------------
# Dataset
# ---------------------------
class PluckCropDataset(Dataset):
    """
    Expects:
      img_dir: contains crop images e.g. xxx.jpg
      label_dir: contains xxx.txt (single line):
                 class x_center y_center w h pluck_x pluck_y
                 pluck_x, pluck_y in [0,1] relative to the crop
    Returns:
      img_tensor (C,H,W), target tensor([pluck_x, pluck_y])
    """
    def __init__(self, img_dir, label_dir, img_size=256, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = sorted([p for p in os.listdir(img_dir) if p.lower().endswith((".jpg",".png",".jpeg"))])
        self.img_size = img_size
        self.transform = transform  # torchvision transform or timm transform
        if self.transform is None:
            # fallback to simple resize + to tensor (but it's better to pass timm transform)
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        label_path = os.path.join(self.label_dir, fname.rsplit('.',1)[0] + ".txt")
        img = Image.open(img_path).convert("RGB")

        # apply transform (expects PIL)
        img_t = self.transform(img)  # tensor HWC->CHW usually if using timm transform returns tensor

        # read label
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        line = open(label_path, "r").read().strip().split()
        # line format: cls x_center y_center w h pluck_x pluck_y
        px = float(line[5])
        py = float(line[6])
        target = torch.tensor([px, py], dtype=torch.float32)

        return img_t, target, fname

# ---------------------------
# Model
# ---------------------------
class HRNetPluckRegressor(nn.Module):
    def __init__(self, backbone_name="hrnet_w18", pretrained=True, head_hidden=512):
        super().__init__()
        # load hrnet backbone from timm (classification model)
        # we'll replace the classifier head with our regression head
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        # backbone will produce a single vector feature per sample (pooling)
        feat_dim = self.backbone.num_features  # timm model property

        # regression head: small MLP -> predict 2 values (x,y)
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, 2),
            nn.Sigmoid()  # force output into [0,1] since labels are normalized
        )

    def forward(self, x):
        feat = self.backbone(x)  # shape (B, feat_dim)
        out = self.regressor(feat)
        return out

# ---------------------------
# Train / Eval loops
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    for imgs, targets, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_mae += torch.abs(preds - targets).mean(dim=1).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    return avg_loss, avg_mae

def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for imgs, targets, _ in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)

            total_loss += loss.item() * imgs.size(0)
            total_mae += torch.abs(preds - targets).mean(dim=1).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    return avg_loss, avg_mae

# ---------------------------
# Utilities
# ---------------------------
def save_checkpoint(state, path):
    torch.save(state, path)

# ---------------------------
# Main
# ---------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    print("Device:", device)

    # prepare timm transform (correct normalization & size)
    dummy_model = timm.create_model(args.backbone, pretrained=False)
    data_config = resolve_data_config({}, model=dummy_model)
    # create_transform returns a torchvision-compatible transform (PIL -> Tensor normalized)
    transform = create_transform(**{**data_config, "is_training": False, "input_size": (3, args.img_size, args.img_size)})
    # For training we can use the same transform but with augmentation turned on if desired (simple augmentation below)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"])
    ])

    # dataset
    img_dir = os.path.join(args.data_dir, "images")
    lbl_dir = os.path.join(args.data_dir, "labels")
    dataset = PluckCropDataset(img_dir, lbl_dir, img_size=args.img_size, transform=train_transform)
    n = len(dataset)
    val_size = int(n * args.val_split)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Fix transforms for val ds (random_split returns Subset - we override dataset.transform)
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}")

    # model
    model = HRNetPluckRegressor(backbone_name=args.backbone, pretrained=True, head_hidden=args.head_hidden).to(device)

    # optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = nn.MSELoss()  # regression MSE; MAE reported as metric

    # optionally resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val_loss = ck.get("best_val", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_mae = validate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"Train loss: {train_loss:.6f} | Train MAE: {train_mae:.6f}")
        print(f"Val   loss: {val_loss:.6f} | Val   MAE: {val_mae:.6f}")

        # checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        ckpt = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val_loss
        }
        save_checkpoint(ckpt, args.out)
        if is_best:
            save_checkpoint(ckpt, args.out.replace(".pth", "_best.pth"))

    print("Training finished. Best val loss:", best_val_loss)

# ---------------------------
# Inference helper
# ---------------------------
def predict_on_crop(model_path, crop_image_path, img_size=256, device="cpu", backbone="hrnet_w18"):
    # load model
    cfg_model = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
    data_config = resolve_data_config({}, model=cfg_model)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"])
    ])
    model = HRNetPluckRegressor(backbone_name=backbone, pretrained=False)
    ck = torch.load(model_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    img = Image.open(crop_image_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)  # (1,C,H,W)
    with torch.no_grad():
        out = model(inp)[0].cpu().numpy()  # normalized px,py
    return float(out[0]), float(out[1])  # px,py in [0,1]

# ---------------------------
# Argparse
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Dataset", help="dataset folder with images/ and labels/")
    parser.add_argument("--backbone", type=str, default="hrnet_w18", help="timm backbone name")
    parser.add_argument("--img_size", type=int, default=640, help="input crop size")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=15)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--head_hidden", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="hrnet_pluck.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=int, default=0, help="gpu device id, set -1 for cpu")
    args = parser.parse_args()
    main(args)
