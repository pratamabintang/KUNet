import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from dataset import LandslideDataset
from model import UNet, FusionMode
from utils.helper import load_config


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
    parser = argparse.ArgumentParser(description="Evaluate and test UNet on Landslide Dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save predictions and metrics")
    parser.add_argument("--save_masks", type=str2bool, default=True, help="Save prediction PNGs (True/False)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")

    # Dataset overrides
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset root directory")
    parser.add_argument("--modalities", type=str, default=None, help="Override modalities (comma-separated, e.g. 'IMAGE,DTM')")
    parser.add_argument("--blacklist_path", type=str, default=None, help="Override blacklist file path")
    parser.add_argument("--size", type=int, default=None, help="Override evaluation spatial target size")

    # Model overrides
    parser.add_argument("--rgb_backbone", type=str, default=None, help="Override RGB backbone (timm)")
    parser.add_argument("--topo_backbone", type=str, default=None, help="Override topography backbone (timm)")
    parser.add_argument("--fusion_mode", type=str, default=None, help="Override fusion mode ('concat')")
    parser.add_argument("--use_kan", type=str2bool, default=None, help="Override KAN decoder flag (True/False)")

    return parser.parse_args()


def calculate_metrics(pred_prob: np.ndarray, gt_binary: np.ndarray, threshold: float = 0.5):
    """
    Computes standard segmentation metrics:
    IoU, Dice/F1, Precision, Recall, MAE, and mAP (Average Precision / PR-AUC).
    """
    pred_bin = (pred_prob >= threshold).astype(np.float32)
    gt = gt_binary.astype(np.float32)

    intersection = (pred_bin * gt).sum()
    union = (pred_bin + gt).clip(0, 1).sum()
    tp = intersection
    fp = (pred_bin * (1 - gt)).sum()
    fn = ((1 - pred_bin) * gt).sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (pred_bin.sum() + gt.sum() + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    mae = np.mean(np.abs(pred_prob - gt))

    gt_flat = gt.flatten().astype(np.int32)
    prob_flat = pred_prob.flatten().astype(np.float32)
    pos_count = int(gt_flat.sum())
    if pos_count == 0:
        ap = 1.0 if np.all(prob_flat < threshold) else 0.0
    elif pos_count == len(gt_flat):
        ap = 1.0 if np.all(prob_flat >= threshold) else 0.0
    else:
        try:
            ap = float(average_precision_score(gt_flat, prob_flat))
        except Exception:
            ap = 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "mae": float(mae),
        "map": float(ap),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # Ensure required sections exist
    config.setdefault("dataset", {})
    config.setdefault("model", {})

    # Apply Dataset overrides
    if args.data_dir is not None:
        config["dataset"]["data_dir"] = args.data_dir
    if args.modalities is not None:
        config["dataset"]["modalities"] = [m.strip().upper() for m in args.modalities.split(",") if m.strip()]
    if args.blacklist_path is not None:
        config["dataset"]["blacklist_path"] = args.blacklist_path
    if args.size is not None:
        config["dataset"]["size"] = args.size

    # Apply Model overrides
    if args.rgb_backbone is not None:
        config["model"]["rgb_backbone"] = args.rgb_backbone
    if args.topo_backbone is not None:
        config["model"]["topo_backbone"] = args.topo_backbone
    if args.fusion_mode is not None:
        config["model"]["fusion_mode"] = args.fusion_mode
    if args.use_kan is not None:
        config["model"]["use_kan"] = args.use_kan

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        run_parent = os.path.dirname(ckpt_dir)
        save_dir = os.path.join(run_parent, f"{args.split}_results")

    probability_dir = os.path.join(save_dir, "probability_maps")
    binary_dir = os.path.join(save_dir, "binary_masks")
    if args.save_masks:
        os.makedirs(probability_dir, exist_ok=True)
        os.makedirs(binary_dir, exist_ok=True)
    print(f"Results and predictions will be saved to: {save_dir}")

    ds_cfg = config["dataset"]
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    data_dir = ds_cfg.get("data_dir", "datasets/landslide")
    blacklist_path = ds_cfg.get("blacklist_path", "datasets/landslide/black_list.txt")
    target_size = ds_cfg.get("size", 512)

    dataset = LandslideDataset(
        data_dir=data_dir,
        split=args.split,
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Evaluating {len(dataset)} samples from [{args.split}] split with modalities: {modalities}")

    # Model initialization
    topo_in_chans = sum(1 for m in modalities if m != "IMAGE")
    m_cfg = config.get("model", {})
    rgb_backbone = m_cfg.get("rgb_backbone", "efficientnet_b5")
    topo_backbone = m_cfg.get("topo_backbone", "efficientnet_b5")
    fusion_mode_str = m_cfg.get("fusion_mode", "concat")
    fusion_mode = FusionMode(fusion_mode_str.lower())
    use_kan = m_cfg.get("use_kan", True)

    print(f"Loading checkpoint weights from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if any("kan." in k for k in state_dict.keys()):
        use_kan = True

    model = UNet(
        rgb_backbone=rgb_backbone,
        pretrained_rgb=False,
        topo_backbone=topo_backbone,
        topo_in_chans=max(1, topo_in_chans),
        pretrained_topo=False,
        fusion=fusion_mode,
        use_kan=use_kan,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating [{args.split}]"):
            images = batch["image"].to(device)
            labels = batch["label"]
            names = batch["name"]

            preds, _, _ = model(images)

            target_size = labels.shape[-2:]
            if preds.shape[-2:] != target_size:
                preds_upsampled = F.interpolate(
                    preds,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                preds_upsampled = preds
            pred_probs = torch.sigmoid(preds_upsampled).cpu().numpy()

            for i in range(len(names)):
                name = names[i]
                prob = pred_probs[i, 0]
                gt = labels[i, 0].numpy()
                m = calculate_metrics(prob, gt, threshold=args.threshold)
                all_metrics.append(m)

                if args.save_masks:
                    probability_img = (prob * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(probability_img).save(
                        os.path.join(probability_dir, f"{name}.png")
                    )

                    binary_img = (prob >= args.threshold).astype(np.uint8) * 255
                    Image.fromarray(binary_img).save(
                        os.path.join(binary_dir, f"{name}.png")
                    )

    avg_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0].keys()
    }

    print("\n" + "=" * 50)
    print(f"EVALUATION METRICS SUMMARY [{args.split.upper()}] (N={len(all_metrics)})")
    print("=" * 50)
    for k, v in avg_metrics.items():
        print(f"  {k.upper():<12}: {v:.4f}")
    print("=" * 50)

    metrics_txt = os.path.join(save_dir, "metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Evaluation on Split: {args.split}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Modalities: {modalities}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        for k, v in avg_metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")

    metrics_json = os.path.join(save_dir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "checkpoint": args.checkpoint,
                "modalities": modalities,
                "threshold": args.threshold,
                "metrics": avg_metrics,
            },
            f,
            indent=2,
        )

    print(f"Metrics saved to:\n  - {metrics_txt}\n  - {metrics_json}")


if __name__ == "__main__":
    main()