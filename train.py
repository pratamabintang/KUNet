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
    parser.add_argument("--encoder_lr", type=float, default=None, help="Override encoder learning rate")
    parser.add_argument("--decoder_lr", type=float, default=None, help="Override decoder/fusion learning rate")
    parser.add_argument("--aux1_weight", type=float, default=None, help="Override weight for aux head 1 (deepest)")
    parser.add_argument("--aux2_weight", type=float, default=None, help="Override weight for aux head 2 (mid)")
    parser.add_argument("--freeze_encoder_bn", type=str2bool, default=None, help="Freeze encoder BatchNorm statistics (True/False)")
    parser.add_argument("--overfit_samples", type=int, default=None, help="Run mini sanity check on N samples")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--min_lr", type=float, default=None, help="Override minimum learning rate")
    parser.add_argument("--save_interval", type=int, default=None, help="Override checkpoint save interval")
    parser.add_argument("--eval_interval", type=int, default=None, help="Override validation evaluation interval")
    parser.add_argument("--amp", type=str2bool, default=None, help="Enable/disable AMP (True/False)")
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["auto", "float16", "fp16", "bfloat16", "bf16"], help="Override AMP precision dtype")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=None, help="Override gradient norm clipping")

    return parser.parse_args()


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates weighted binary cross-entropy and weighted IoU loss.
    Always operates in float32 to prevent float16 overflow on 512x512 spatial sums
    (262,144 pixels strictly exceeds float16 maximum finite value of 65,504).
    """
    pred = pred.float()
    mask = mask.float()

    weit = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-7)

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (wbce + wiou).mean()


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    aux1_weight: float = 0.4,
    aux2_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Evaluates model on validation set with separate logging for main and auxiliary losses,
    and computes dataset-level (global) intersection-over-union and Dice score.
    """
    model.eval()
    total_main_loss = 0.0
    total_aux1_loss = 0.0
    total_aux2_loss = 0.0
    total_weighted_loss = 0.0
    batch_count = 0

    total_intersection = 0.0
    total_union = 0.0
    total_pred_pos = 0.0
    total_target_pos = 0.0
    total_pixels = 0
    empty_mask_batches = 0

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

            # Compute validation losses in float32 for complete numerical stability
            loss_main = structure_loss(out, target)
            loss_aux1 = structure_loss(out1, target)
            loss_aux2 = structure_loss(out2, target)
            loss = loss_main + (aux1_weight * loss_aux1) + (aux2_weight * loss_aux2)

            total_main_loss += loss_main.item()
            total_aux1_loss += loss_aux1.item()
            total_aux2_loss += loss_aux2.item()
            total_weighted_loss += loss.item()
            batch_count += 1

            # Threshold binary prediction from primary head
            pred_bin = (torch.sigmoid(out) >= 0.5).float()
            target_bin = (target >= 0.5).float()

            inter = (pred_bin * target_bin).sum().item()
            union = ((pred_bin + target_bin) > 0).float().sum().item()
            pred_pos = pred_bin.sum().item()
            target_pos = target_bin.sum().item()

            total_intersection += inter
            total_union += union
            total_pred_pos += pred_pos
            total_target_pos += target_pos
            total_pixels += target.numel()

            if target_pos == 0:
                empty_mask_batches += 1

    dataset_iou = total_intersection / max(total_union, 1e-7)
    dataset_dice = (2.0 * total_intersection) / max(total_pred_pos + total_target_pos, 1e-7)
    positive_ratio = total_target_pos / max(total_pixels, 1)
    empty_mask_ratio = empty_mask_batches / max(batch_count, 1)

    return {
        "val_main_loss": total_main_loss / max(1, batch_count),
        "val_aux1_loss": total_aux1_loss / max(1, batch_count),
        "val_aux2_loss": total_aux2_loss / max(1, batch_count),
        "val_total_loss": total_weighted_loss / max(1, batch_count),
        "val_iou": float(dataset_iou),
        "val_dice": float(dataset_dice),
        "positive_ratio": float(positive_ratio),
        "empty_mask_ratio": float(empty_mask_ratio),
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
    if args.encoder_lr is not None:
        config["training"]["encoder_lr"] = args.encoder_lr
    if args.decoder_lr is not None:
        config["training"]["decoder_lr"] = args.decoder_lr
    if args.aux1_weight is not None:
        config["training"]["aux1_weight"] = args.aux1_weight
    if args.aux2_weight is not None:
        config["training"]["aux2_weight"] = args.aux2_weight
    if args.freeze_encoder_bn is not None:
        config["training"]["freeze_encoder_bn"] = args.freeze_encoder_bn
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

    if args.overfit_samples is not None and args.overfit_samples > 0:
        sample_indices = list(range(min(args.overfit_samples, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, sample_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, sample_indices)
        logger.info(f"[*] OVERFIT SANITY MODE: Dataset subsetted to {len(sample_indices)} sample(s).")

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
    rgb_backbone = m_cfg.get("rgb_backbone", "efficientnet_b5")
    pretrained_rgb = m_cfg.get("pretrained_rgb", True)
    topo_backbone = m_cfg.get("topo_backbone", "efficientnet_b5")
    pretrained_topo = m_cfg.get("pretrained_topo", False)
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

    # Optimizer & Scheduler with Differential Learning Rates
    t_cfg = config["training"]
    epochs = t_cfg.get("epochs", 20)
    base_lr = float(t_cfg.get("lr", 1.0e-4))
    encoder_lr = float(t_cfg.get("encoder_lr", base_lr * 0.1 if base_lr >= 1e-4 else 1.0e-5))
    decoder_lr = float(t_cfg.get("decoder_lr", base_lr))
    aux1_weight = float(t_cfg.get("aux1_weight", 0.4))
    aux2_weight = float(t_cfg.get("aux2_weight", 0.2))
    freeze_encoder_bn = bool(t_cfg.get("freeze_encoder_bn", False))
    weight_decay = float(t_cfg.get("weight_decay", 5.0e-4))
    min_lr = float(t_cfg.get("min_lr", 1.0e-7))
    save_interval = int(t_cfg.get("save_interval", 10))
    eval_interval = int(t_cfg.get("eval_interval", 1))
    amp_enabled = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    grad_accum_steps = max(1, int(t_cfg.get("grad_accum_steps", 1)))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))

    if freeze_encoder_bn:
        logger.info("Freezing BatchNorm running statistics on RGB and Topo encoders (freeze_encoder_bn=True)...")
        model.freeze_encoder_bn()

    # Differential Learning Rate Parameter Groups
    rgb_params = [p for p in model.rgb_encoder.parameters() if p.requires_grad]
    topo_params = [p for p in model.topo_encoder.parameters() if p.requires_grad]
    encoder_param_ids = set(id(p) for p in rgb_params + topo_params)
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids and p.requires_grad]
    trainable_params = rgb_params + topo_params + decoder_params

    param_groups = [
        {"params": rgb_params, "lr": encoder_lr, "name": "rgb_encoder"},
        {"params": topo_params, "lr": encoder_lr, "name": "topo_encoder"},
        {"params": decoder_params, "lr": decoder_lr, "name": "decoder_fusion_heads"},
    ]

    logger.info(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"Optimizer Parameter Groups:")
    logger.info(f"  [1] rgb_encoder:          {sum(p.numel() for p in rgb_params):,} params | lr={encoder_lr:.2e}")
    logger.info(f"  [2] topo_encoder:         {sum(p.numel() for p in topo_params):,} params | lr={encoder_lr:.2e}")
    logger.info(f"  [3] decoder_fusion_heads: {sum(p.numel() for p in decoder_params):,} params | lr={decoder_lr:.2e}")
    logger.info(f"Deep Supervision Weights: main=1.0, aux1={aux1_weight}, aux2={aux2_weight}")

    optimizer = opt.AdamW(param_groups, weight_decay=weight_decay)
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
    best_val_main_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_encoder_bn:
            model.freeze_encoder_bn()

        epoch_main_loss = 0.0
        epoch_aux1_loss = 0.0
        epoch_aux2_loss = 0.0
        epoch_total_loss = 0.0
        epoch_grad_norm = 0.0
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

            # Compute losses in float32 for complete numerical stability under AMP
            loss_main = structure_loss(out, target)
            loss_aux1 = structure_loss(out1, target)
            loss_aux2 = structure_loss(out2, target)
            loss = loss_main + (aux1_weight * loss_aux1) + (aux2_weight * loss_aux2)

            group_start = (i // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, len(train_loader) - group_start)
            scaler.scale(loss / group_size).backward()

            grad_norm = 0.0
            should_step = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == len(train_loader))
            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_main_loss += loss_main.item()
            epoch_aux1_loss += loss_aux1.item()
            epoch_aux2_loss += loss_aux2.item()
            epoch_total_loss += loss.item()
            epoch_grad_norm += grad_norm
            step_count += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                cur_enc_lr = optimizer.param_groups[0]["lr"]
                cur_dec_lr = optimizer.param_groups[2]["lr"]
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Step [{i+1}/{len(train_loader)}] - "
                    f"Total: {loss.item():.4f} (main={loss_main.item():.4f}, aux1={loss_aux1.item():.4f}, aux2={loss_aux2.item():.4f}) - "
                    f"GradNorm: {grad_norm:.4f} - LR: enc={cur_enc_lr:.2e}, dec={cur_dec_lr:.2e}"
                )

        scheduler.step()
        avg_train_main = epoch_main_loss / max(1, step_count)
        avg_train_aux1 = epoch_aux1_loss / max(1, step_count)
        avg_train_aux2 = epoch_aux2_loss / max(1, step_count)
        avg_train_total = epoch_total_loss / max(1, step_count)
        avg_train_grad_norm = epoch_grad_norm / max(1, step_count)

        logger.info(
            f"Epoch [{epoch}/{epochs}] Finished - Train: "
            f"main={avg_train_main:.4f}, aux1={avg_train_aux1:.4f}, aux2={avg_train_aux2:.4f}, "
            f"total={avg_train_total:.4f}, grad_norm={avg_train_grad_norm:.4f}"
        )

        val_metrics = None
        if epoch % eval_interval == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                aux1_weight=aux1_weight,
                aux2_weight=aux2_weight,
            )
            logger.info(
                f"Validation Epoch [{epoch}/{epochs}] - "
                f"Val: main={val_metrics['val_main_loss']:.4f}, aux1={val_metrics['val_aux1_loss']:.4f}, "
                f"aux2={val_metrics['val_aux2_loss']:.4f}, total={val_metrics['val_total_loss']:.4f} | "
                f"IoU={val_metrics['val_iou']:.4f}, Dice={val_metrics['val_dice']:.4f} | "
                f"PosRatio={val_metrics['positive_ratio']:.4f}, EmptyRatio={val_metrics['empty_mask_ratio']:.4f}"
            )

            # Checkpoint on best validation IoU
            if val_metrics["val_iou"] > best_val_iou:
                best_val_iou = val_metrics["val_iou"]
                best_ckpt_path = os.path.join(checkpoints_dir, "best.pth")
                best_checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_main_loss": val_metrics["val_main_loss"],
                    "val_iou": val_metrics["val_iou"],
                    "val_dice": val_metrics["val_dice"],
                    "config": config,
                }
                torch.save(best_checkpoint_data, best_ckpt_path)
                logger.info(f"[*] New best validation IoU ({best_val_iou:.4f})! Saved to {best_ckpt_path}")

            # Checkpoint on best validation main loss
            if val_metrics["val_main_loss"] < best_val_main_loss:
                best_val_main_loss = val_metrics["val_main_loss"]
                best_loss_ckpt_path = os.path.join(checkpoints_dir, "best_loss.pth")
                best_loss_checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_main_loss": val_metrics["val_main_loss"],
                    "val_iou": val_metrics["val_iou"],
                    "val_dice": val_metrics["val_dice"],
                    "config": config,
                }
                torch.save(best_loss_checkpoint_data, best_loss_ckpt_path)
                logger.info(f"[*] New best validation main loss ({best_val_main_loss:.4f})! Saved to {best_loss_ckpt_path}")

        latest_ckpt_path = os.path.join(checkpoints_dir, "latest.pth")
        latest_checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_main_loss": val_metrics["val_main_loss"] if val_metrics else None,
            "val_iou": val_metrics["val_iou"] if val_metrics else None,
            "val_dice": val_metrics["val_dice"] if val_metrics else None,
            "config": config,
        }
        torch.save(latest_checkpoint_data, latest_ckpt_path)

        if epoch % save_interval == 0 or epoch == epochs:
            snap_path = os.path.join(checkpoints_dir, f"epoch_{epoch}.pth")
            torch.save(latest_checkpoint_data, snap_path)
            logger.info(f"Saved snapshot to {snap_path}")

    total_time = (time.time() - start_time) / 60
    logger.info(f"=== Training Completed in {total_time:.2f} minutes ===")
    logger.info(f"Best Validation IoU: {best_val_iou:.4f} | Best Validation Main Loss: {best_val_main_loss:.4f}")
    logger.info(f"Artifacts and checkpoints are preserved at: {run_dir}")

if __name__ == "__main__":
    main()