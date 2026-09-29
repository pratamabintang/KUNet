from typing import Dict, Any, List, Optional
import os
import argparse
import yaml
from datetime import datetime
import time
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import LandslideDataset
from model import UNet, FusionMode
from utils.helper import seed_everything, load_config, setup_logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet with Dual-Branch Architecture")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration file")

    # Experiment overrides
    parser.add_argument("--run_name", type=str, default=None, help="Override experiment/run name")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--runs_dir", type=str, default=None, help="Override runs output directory")

    # Dataset overrides
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset root directory")
    parser.add_argument("--modalities", type=str, default=None, help="Override modalities (comma-separated, e.g. 'IMAGE,DTM')")
    parser.add_argument("--blacklist_path", type=str, default=None, help="Override blacklist file path")
    parser.add_argument("--size", type=int, default=None, help="Override spatial target size")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers")

    # Model overrides
    parser.add_argument("--rgb_backbone", type=str, default=None, help="Override RGB backbone (timm)")
    parser.add_argument("--pretrained_rgb", type=str2bool, default=None, help="Override pretrained RGB backbone flag (True/False)")
    parser.add_argument("--topo_backbone", type=str, default=None, help="Override topography backbone (timm)")
    parser.add_argument("--pretrained_topo", type=str2bool, default=None, help="Override pretrained topo backbone flag (True/False)")
    parser.add_argument("--fusion_mode", type=str, default=None, help="Override fusion mode ('concat')")
    parser.add_argument("--use_kan", type=str2bool, default=None, help="Enable/disable KAN decoder (True/False)")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--min_lr", type=float, default=None, help="Override minimum learning rate")
    parser.add_argument("--save_interval", type=int, default=None, help="Override checkpoint save interval")
    parser.add_argument("--eval_interval", type=int, default=None, help="Override validation evaluation interval")
    parser.add_argument("--amp", type=str2bool, default=None, help="Enable/disable AMP (True/False)")
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["auto", "float16", "fp16", "bfloat16", "bf16"], help="Override AMP precision dtype")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=None, help="Override gradient norm clipping")

    return parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def compute_iou(pred: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    inter = (pred_bin * mask).sum().item()
    union = (pred_bin + mask).clamp(0, 1).sum().item()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / (union + 1e-7)


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, amp_enabled: bool = False, amp_dtype: torch.dtype = torch.float16) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                out, out1, out2 = model(image)
                loss0 = structure_loss(out, target)
                loss1 = structure_loss(out1, target)
                loss2 = structure_loss(out2, target)
                loss = loss0 + loss1 + loss2

            total_loss += loss.item()
            total_iou += compute_iou(out, target)
            count += 1

    return {
        "val_loss": total_loss / max(1, count),
        "val_iou": total_iou / max(1, count),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # Ensure required sections exist
    config.setdefault("experiment", {})
    config.setdefault("dataset", {})
    config.setdefault("model", {})
    config.setdefault("training", {})

    # Experiment overrides
    if args.run_name is not None:
        config["experiment"]["name"] = args.run_name
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.runs_dir is not None:
        config["experiment"]["runs_dir"] = args.runs_dir

    # Dataset overrides
    if args.data_dir is not None:
        config["dataset"]["data_dir"] = args.data_dir
    if args.modalities is not None:
        config["dataset"]["modalities"] = [m.strip().upper() for m in args.modalities.split(",") if m.strip()]
    if args.blacklist_path is not None:
        config["dataset"]["blacklist_path"] = args.blacklist_path
    if args.size is not None:
        config["dataset"]["size"] = args.size
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["dataset"]["num_workers"] = args.num_workers

    # Model overrides
    if args.rgb_backbone is not None:
        config["model"]["rgb_backbone"] = args.rgb_backbone
    if args.pretrained_rgb is not None:
        config["model"]["pretrained_rgb"] = args.pretrained_rgb
    if args.topo_backbone is not None:
        config["model"]["topo_backbone"] = args.topo_backbone
    if args.pretrained_topo is not None:
        config["model"]["pretrained_topo"] = args.pretrained_topo
    if args.fusion_mode is not None:
        config["model"]["fusion_mode"] = args.fusion_mode
    if args.use_kan is not None:
        config["model"]["use_kan"] = args.use_kan

    # Training overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.min_lr is not None:
        config["training"]["min_lr"] = args.min_lr
    if args.save_interval is not None:
        config["training"]["save_interval"] = args.save_interval
    if args.eval_interval is not None:
        config["training"]["eval_interval"] = args.eval_interval
    if args.amp is not None:
        config["training"]["amp"] = args.amp
    if args.amp_dtype is not None:
        config["training"]["amp_dtype"] = args.amp_dtype
    if args.grad_accum_steps is not None:
        config["training"]["grad_accum_steps"] = args.grad_accum_steps
    if args.grad_clip is not None:
        config["training"]["grad_clip"] = args.grad_clip

    seed = config["experiment"].get("seed", 1024)
    seed_everything(seed)

    runs_dir = config["experiment"].get("runs_dir", "runs")
    exp_name = config["experiment"].get("name", "unet")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{exp_name}_{timestamp}")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    saved_config_path = os.path.join(run_dir, "config.yaml")
    with open(saved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger = setup_logger(os.path.join(run_dir, "train.log"))
    logger.info(f"=== Starting Training Session ===")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Run Output Directory: {run_dir}")
    logger.info(f"Active Config saved to: {saved_config_path}")

    ds_cfg = config["dataset"]
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    data_dir = ds_cfg.get("data_dir", "datasets/landslide")
    blacklist_path = ds_cfg.get("blacklist_path", "datasets/landslide/black_list.txt")
    target_size = ds_cfg.get("size", 512)
    batch_size = ds_cfg.get("batch_size", 12)
    num_workers = ds_cfg.get("num_workers", 4)

    logger.info(f"Selected Modalities: {modalities}")
    logger.info(f"Spatial Target Size: {target_size}x{target_size}, Batch Size: {batch_size}")

    train_dataset = LandslideDataset(
        data_dir=data_dir,
        split="train",
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
    )
    val_dataset = LandslideDataset(
        data_dir=data_dir,
        split="val",
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device}")

    train_workers = num_workers if (device.type == "cuda" and os.name != "nt") else min(num_workers, 2)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_dataset) > batch_size else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    topo_in_chans = sum(1 for m in modalities if m != "IMAGE")
    logger.info(f"Derived Topography Input Channels: {topo_in_chans}")

    m_cfg = config["model"]
    rgb_backbone = m_cfg.get("rgb_backbone", "convnext_tiny")
    pretrained_rgb = m_cfg.get("pretrained_rgb", True)
    topo_backbone = m_cfg.get("topo_backbone", "convnext_tiny")
    pretrained_topo = m_cfg.get("pretrained_topo", True)
    fusion_mode_str = m_cfg.get("fusion_mode", "concat")
    fusion_mode = FusionMode(fusion_mode_str.lower())
    use_kan = m_cfg.get("use_kan", True)

    logger.info(f"Initializing UNet (RGB: {rgb_backbone} [pretrained={pretrained_rgb}], Topo: {topo_backbone} [pretrained={pretrained_topo}], Fusion: {fusion_mode.value}, KAN Decoder: {use_kan})...")
    model = UNet(
        rgb_backbone=rgb_backbone,
        pretrained_rgb=pretrained_rgb,
        topo_backbone=topo_backbone,
        topo_in_chans=max(1, topo_in_chans),
        pretrained_topo=pretrained_topo,
        fusion=fusion_mode,
        use_kan=use_kan
    )
    model.to(device)

    # Optimizer & Scheduler
    t_cfg = config["training"]
    epochs = t_cfg.get("epochs", 20)
    lr = t_cfg.get("lr", 1.0e-3)
    weight_decay = t_cfg.get("weight_decay", 5.0e-4)
    min_lr = t_cfg.get("min_lr", 1.0e-7)
    save_interval = t_cfg.get("save_interval", 10)
    eval_interval = t_cfg.get("eval_interval", 1)
    amp_enabled = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    grad_accum_steps = max(1, int(t_cfg.get("grad_accum_steps", 1)))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = opt.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    amp_dtype_str = str(t_cfg.get("amp_dtype", "auto")).lower()
    if amp_dtype_str in ("bfloat16", "bf16"):
        amp_dtype = torch.bfloat16
    elif amp_dtype_str in ("float16", "fp16"):
        amp_dtype = torch.float16
    else:
        if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    use_scaler = amp_enabled and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    logger.info(
        f"AMP: {amp_enabled} (dtype: {amp_dtype}) | Scaler: {use_scaler} "
        f"| Gradient accumulation: {grad_accum_steps} "
        f"| Effective batch size: {batch_size * grad_accum_steps} "
        f"| Gradient clip: {grad_clip}"
    )

    best_val_iou = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                out, out1, out2 = model(image)
                loss0 = structure_loss(out, target)
                loss1 = structure_loss(out1, target)
                loss2 = structure_loss(out2, target)
                loss = loss0 + loss1 + loss2

            group_start = (i // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, len(train_loader) - group_start)
            scaler.scale(loss / group_size).backward()

            should_step = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == len(train_loader))
            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            step_count += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Step [{i+1}/{len(train_loader)}] - "
                    f"Batch Loss: {loss.item():.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, step_count)
        logger.info(f"Epoch [{epoch}/{epochs}] Finished - Average Train Loss: {avg_train_loss:.4f}")

        if epoch % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
            val_loss = val_metrics["val_loss"]
            val_iou = val_metrics["val_iou"]
            logger.info(f"Validation - Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_ckpt_path = os.path.join(checkpoints_dir, "best.pth")
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"[*] New best validation IoU ({best_val_iou:.4f})! Saved to {best_ckpt_path}")

        latest_ckpt_path = os.path.join(checkpoints_dir, "latest.pth")
        torch.save(model.state_dict(), latest_ckpt_path)

        if epoch % save_interval == 0 or epoch == epochs:
            snap_path = os.path.join(checkpoints_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), snap_path)
            logger.info(f"Saved snapshot to {snap_path}")

    total_time = (time.time() - start_time) / 60
    logger.info(f"=== Training Completed in {total_time:.2f} minutes ===")
    logger.info(f"Best Validation IoU: {best_val_iou:.4f}")
    logger.info(f"Artifacts and checkpoints are preserved at: {run_dir}")

if __name__ == "__main__":
    main()