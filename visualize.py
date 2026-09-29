import os
import sys
import json
import yaml
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from dataset import LandslideDataset
from model import UNet, FusionMode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 4-panel visualizations: Original Image, Ground Truth, Prediction, and Confidence Map"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint file for inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to existing test output folder containing 'probability_maps/' and/or 'binary_masks/'",
    )

    parser.add_argument("--data_dir", type=str, default="datasets/landslide", help="Path to dataset root folder")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated comparison plots")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize (-1 for all samples)",
    )
    parser.add_argument(
        "--sample_names",
        type=str,
        default=None,
        help="Comma-separated list of specific sample base names to visualize (e.g. 'sample_1,sample_2')",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap for Confidence Map (e.g. 'jet', 'turbo', 'magma', 'viridis')",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Add semi-transparent red prediction overlay on the original image",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI resolution for saved figures")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sample_metrics(pred_prob: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes sample-level IoU, Dice, and AP."""
    pred_bin = (pred_prob >= threshold).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)

    inter = (pred_bin * gt_bin).sum()
    union = (pred_bin + gt_bin).clip(0, 1).sum()
    iou = (inter + 1e-7) / (union + 1e-7)
    dice = (2.0 * inter + 1e-7) / (pred_bin.sum() + gt_bin.sum() + 1e-7)

    gt_flat = gt_bin.flatten().astype(np.int32)
    prob_flat = pred_prob.flatten().astype(np.float32)
    if gt_flat.sum() == 0:
        ap = 1.0 if np.all(prob_flat < threshold) else 0.0
    elif gt_flat.sum() == len(gt_flat):
        ap = 1.0 if np.all(prob_flat >= threshold) else 0.0
    else:
        try:
            ap = float(average_precision_score(gt_flat, prob_flat))
        except Exception:
            ap = 0.0

    return {"iou": float(iou), "dice": float(dice), "ap": float(ap)}


def plot_4panel(
    original_rgb: np.ndarray,
    ground_truth: np.ndarray,
    prediction_binary: np.ndarray,
    confidence_map: np.ndarray,
    sample_name: str,
    output_path: str,
    threshold: float = 0.5,
    colormap: str = "jet",
    metrics: Optional[Dict[str, float]] = None,
    overlay: bool = False,
    dpi: int = 200,
):
    """
    Renders and saves a 4-panel visualization:
    [ Gambar Asli | Ground Truth | Prediction Mask | Confidence Map ]
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")

    if metrics is not None:
        title_text = (
            f"Sample: {sample_name} | Threshold: {threshold:.2f} | "
            f"IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f} | AP: {metrics['ap']:.4f}"
        )
    else:
        title_text = f"Sample: {sample_name} | Threshold: {threshold:.2f}"
    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98)

    if overlay:
        overlay_rgb = original_rgb.copy()
        pred_mask = prediction_binary > 0
        overlay_rgb[pred_mask, 0] = np.clip(overlay_rgb[pred_mask, 0] * 0.5 + 128, 0, 255)
        axes[0].imshow(overlay_rgb.astype(np.uint8))
        axes[0].set_title("Original Image (RGB + Pred Overlay)", fontsize=12, pad=10)
    else:
        axes[0].imshow(original_rgb.astype(np.uint8))
        axes[0].set_title("Original Image (RGB)", fontsize=12, pad=10)
    axes[0].axis("off")

    gt_display = (ground_truth > 0).astype(np.float32)
    axes[1].imshow(gt_display, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth (GT)", fontsize=12, pad=10)
    axes[1].axis("off")

    pred_display = (prediction_binary > 0).astype(np.float32)
    axes[2].imshow(pred_display, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (Binary $\\geq$ {threshold:.2f})", fontsize=12, pad=10)
    axes[2].axis("off")

    conf_im = axes[3].imshow(confidence_map, cmap=colormap, vmin=0.0, vmax=1.0)
    axes[3].set_title(f"Confidence Map ({colormap.capitalize()})", fontsize=12, pad=10)
    axes[3].axis("off")

    cbar = fig.colorbar(conf_im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Confidence / Probability", fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.86)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_from_results(args):
    """Generates visualizations by reading saved prediction maps from results_dir."""
    print(f"\n[Mode: Offline Results] Reading from: {args.results_dir}")
    prob_dir = os.path.join(args.results_dir, "probability_maps")
    bin_dir = os.path.join(args.results_dir, "binary_masks")

    if not os.path.exists(prob_dir):
        raise FileNotFoundError(f"Missing probability_maps directory in {args.results_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.results_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    split_dir = os.path.join(args.data_dir, args.split)
    img_dir = os.path.join(split_dir, "IMAGE")
    label_dir = os.path.join(split_dir, "LABEL")

    available_probs = sorted([f for f in os.listdir(prob_dir) if f.endswith(".png") or f.endswith(".tif")])
    if len(available_probs) == 0:
        raise RuntimeError(f"No prediction maps found in {prob_dir}")

    if args.sample_names:
        selected_names = [s.strip() for s in args.sample_names.split(",") if s.strip()]
        sample_files = [f"{s}.png" if not f.endswith((".png", ".tif")) else s for s in selected_names]
    else:
        sample_files = available_probs
        if args.num_samples > 0:
            sample_files = sample_files[: args.num_samples]

    print(f"Visualizing {len(sample_files)} sample(s) -> Output directory: {output_dir}")

    for file_name in tqdm(sample_files, desc="Rendering Visualizations"):
        base_name, _ = os.path.splitext(file_name)

        prob_path = os.path.join(prob_dir, file_name)
        if not os.path.exists(prob_path):
            prob_path = os.path.join(prob_dir, base_name + ".png")
            if not os.path.exists(prob_path):
                continue
        prob_img = Image.open(prob_path)
        prob_arr = np.array(prob_img, dtype=np.float32)
        if prob_arr.max() > 1.0:
            prob_arr = prob_arr / 255.0

        bin_path = os.path.join(bin_dir, base_name + ".png")
        if os.path.exists(bin_path):
            bin_arr = np.array(Image.open(bin_path)) > 0
        else:
            bin_arr = prob_arr >= args.threshold

        img_path = os.path.join(img_dir, base_name + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, base_name + ".tif")
        if os.path.exists(img_path):
            orig_rgb = np.array(Image.open(img_path).convert("RGB"))
        else:
            # Fallback placeholder if original image not found
            orig_rgb = np.zeros((*prob_arr.shape, 3), dtype=np.uint8)

        lbl_path = os.path.join(label_dir, base_name + ".png")
        if not os.path.exists(lbl_path):
            lbl_path = os.path.join(label_dir, base_name + ".tif")
        if os.path.exists(lbl_path):
            gt_arr = np.array(Image.open(lbl_path))
            metrics = compute_sample_metrics(prob_arr, gt_arr, threshold=args.threshold)
        else:
            gt_arr = np.zeros_like(prob_arr)
            metrics = None

        out_file = os.path.join(output_dir, f"{base_name}_vis.png")
        plot_4panel(
            original_rgb=orig_rgb,
            ground_truth=gt_arr,
            prediction_binary=bin_arr,
            confidence_map=prob_arr,
            sample_name=base_name,
            output_path=out_file,
            threshold=args.threshold,
            colormap=args.colormap,
            metrics=metrics,
            overlay=args.overlay,
            dpi=args.dpi,
        )

    print(f"\n[DONE] All visualizations saved to: {output_dir}")


def visualize_from_model(args):
    """Runs inference directly using model checkpoint and generates 4-panel visualizations."""
    print(f"\n[Mode: Model Inference] Checkpoint: {args.checkpoint}")
    config = load_config(args.config)
    ds_cfg = config["dataset"]
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    data_dir = args.data_dir or ds_cfg.get("data_dir", "datasets/landslide")
    blacklist_path = ds_cfg.get("blacklist_path", "datasets/landslide/black_list.txt")
    target_size = ds_cfg.get("size", 352)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} | Spatial Target Size: {target_size}x{target_size}")

    dataset = LandslideDataset(
        data_dir=data_dir,
        split=args.split,
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
    )

    topo_in_chans = sum(1 for m in modalities if m != "IMAGE")
    m_cfg = config.get("model", {})
    rgb_backbone = m_cfg.get("rgb_backbone", "convnext_tiny")
    topo_backbone = m_cfg.get("topo_backbone", "convnext_tiny")
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
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    output_dir = args.output_dir
    if output_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        run_parent = os.path.dirname(ckpt_dir)
        output_dir = os.path.join(run_parent, f"{args.split}_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    total_samples = len(dataset)
    if args.sample_names:
        requested = set(s.strip() for s in args.sample_names.split(","))
        indices = [i for i, name in enumerate(dataset.samples) if name in requested]
    else:
        limit = total_samples if args.num_samples <= 0 else min(args.num_samples, total_samples)
        indices = list(range(limit))

    print(f"Rendering {len(indices)} sample(s) -> Output directory: {output_dir}")

    split_dir = os.path.join(data_dir, args.split)
    img_dir = os.path.join(split_dir, "IMAGE")

    with torch.no_grad():
        for idx in tqdm(indices, desc="Inference & Rendering"):
            sample = dataset[idx]
            name = sample["name"]
            image_tensor = sample["image"].unsqueeze(0).to(device)  # (1, C, H, W)
            label_tensor = sample["label"]  # (1, H, W)

            preds, _, _ = model(image_tensor)
            if preds.shape[-2:] != label_tensor.shape[-2:]:
                preds = F.interpolate(
                    preds,
                    size=label_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            prob_map = torch.sigmoid(preds)[0, 0].cpu().numpy()  # (H, W)
            pred_bin = prob_map >= args.threshold
            gt_arr = label_tensor[0].numpy()

            orig_img_path = os.path.join(img_dir, f"{name}.png")
            if not os.path.exists(orig_img_path):
                orig_img_path = os.path.join(img_dir, f"{name}.tif")
            if os.path.exists(orig_img_path):
                raw_rgb = np.array(Image.open(orig_img_path).convert("RGB"))
                if raw_rgb.shape[:2] != prob_map.shape:
                    raw_rgb = np.array(Image.fromarray(raw_rgb).resize((prob_map.shape[1], prob_map.shape[0])))
            else:
                rgb_t = sample["image"][:3].cpu().numpy().transpose(1, 2, 0)
                mean = np.array(LandslideDataset.IMAGENET_MEAN)
                std = np.array(LandslideDataset.IMAGENET_STD)
                raw_rgb = np.clip((rgb_t * std + mean) * 255.0, 0, 255).astype(np.uint8)

            metrics = compute_sample_metrics(prob_map, gt_arr, threshold=args.threshold)
            out_file = os.path.join(output_dir, f"{name}_vis.png")

            plot_4panel(
                original_rgb=raw_rgb,
                ground_truth=gt_arr,
                prediction_binary=pred_bin,
                confidence_map=prob_map,
                sample_name=name,
                output_path=out_file,
                threshold=args.threshold,
                colormap=args.colormap,
                metrics=metrics,
                overlay=args.overlay,
                dpi=args.dpi,
            )

    print(f"\n[DONE] All visualizations saved to: {output_dir}")


def main():
    args = parse_args()
    if args.results_dir is not None:
        visualize_from_results(args)
    elif args.checkpoint is not None:
        visualize_from_model(args)
    else:
        # Check if default test results folder exists
        default_res = "test_results"
        if os.path.exists(default_res):
            args.results_dir = default_res
            visualize_from_results(args)
        else:
            print("ERROR: Please specify either --checkpoint <path.pth> or --results_dir <path_to_results>")
            sys.exit(1)


if __name__ == "__main__":
    main()
