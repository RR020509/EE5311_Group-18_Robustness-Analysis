import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter

MNIST_CLASSES = [str(i) for i in range(10)]

OUTPUT_PREFIX = "robust_mlp_mnist"
CHECKPOINT_FILENAME = f"{OUTPUT_PREFIX}_finetuned.pt"
REPORT_FILENAME = f"{OUTPUT_PREFIX}_finetune_report.json"


class SimpleMLP(nn.Module):
    # Initialize the MLP layers for MNIST classification.
    def __init__(self, input_dim: int = 28 * 28, hidden_dim1: int = 256, hidden_dim2: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, num_classes),
        )

    # Run a forward pass to produce class logits.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Parse command-line options for training and evaluation.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SimpleMLP on MNIST and evaluate test accuracy."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--download-dataset", action="store_true", default=True)
    parser.add_argument("--no-download-dataset", dest="download_dataset", action="store_false")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-train", action="store_true", default=False)
    parser.add_argument("--no-force-train", dest="force_train", action="store_false")
    parser.add_argument("--run-train", action="store_true", default=False)
    parser.add_argument("--run-sensitivity", action="store_true", default=False)
    parser.add_argument("--run-robustness", action="store_true", default=False)
    parser.add_argument("--run-adversarial", action="store_true", default=False)
    parser.add_argument("--run-sampling", action="store_true", default=False)
    # Sensitivity analysis options:
    parser.add_argument("--image-index", type=int, default=35)
    parser.add_argument("--topk-jacobian", type=int, default=3)
    parser.add_argument("--ig-steps", type=int, default=100)
    parser.add_argument("--smoothgrad-samples", type=int, default=100)
    parser.add_argument("--smoothgrad-noise-std", type=float, default=0.2)
    # Analytical robustness options (Section 3):
    parser.add_argument("--robustness-eps", type=float, default=0.25)
    parser.add_argument("--robustness-trials", type=int, default=30)
    parser.add_argument("--robustness-eval-samples", type=int, default=20)
    parser.add_argument("--robustness-sweep-eps", type=str, default="0.05,0.1,0.2,0.3,0.4")
    parser.add_argument("--robustness-dist-samples", type=int, default=20)
    parser.add_argument("--robustness-scatter-trials", type=int, default=5)

    # Section 5
    # Random perturbation analysis parameters
    parser.add_argument("--noise-type", type=str, default="gaussian",
                        choices=["gaussian", "uniform", "salt_pepper", "speckle", "all"])
    parser.add_argument('--sample-strategy', type=str, default='correct',
                        choices=['correct', 'random', 'stratified', 'hard'])
    parser.add_argument("--min-noise-std", type=float, default=0.0)
    parser.add_argument("--max-noise-std", type=float, default=0.3)
    parser.add_argument("--num-noise-levels", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-monte-carlo", type=int, default=1000)
    return parser.parse_args()


# Validate CLI arguments for safer training and sensitivity runs.
def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.topk_jacobian < 1:
        raise ValueError("--topk-jacobian must be >= 1")
    if args.ig_steps < 1:
        raise ValueError("--ig-steps must be >= 1")
    if args.smoothgrad_samples < 1:
        raise ValueError("--smoothgrad-samples must be >= 1")
    if args.smoothgrad_noise_std < 0:
        raise ValueError("--smoothgrad-noise-std must be >= 0")
    if args.robustness_eps <= 0:
        raise ValueError("--robustness-eps must be > 0")
    if args.robustness_trials < 1:
        raise ValueError("--robustness-trials must be >= 1")
    if args.robustness_eval_samples < 1:
        raise ValueError("--robustness-eval-samples must be >= 1")
    if args.robustness_dist_samples < 1:
        raise ValueError("--robustness-dist-samples must be >= 1")
    if args.robustness_scatter_trials < 1:
        raise ValueError("--robustness-scatter-trials must be >= 1")

    # Validate random perturbation analysis parameters
    if args.min_noise_std < 0:
        raise ValueError("--min-noise-std must be >= 0")
    if args.max_noise_std < args.min_noise_std:
        raise ValueError("--max-noise-std must be >= min-noise-std")
    if args.num_noise_levels < 2:
        raise ValueError("--num-noise-levels must be >= 2")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.num_monte_carlo < 1:
        raise ValueError("--num-monte-carlo must be >= 1")


def parse_float_csv(csv_values: str) -> List[float]:
    values = [float(v.strip()) for v in csv_values.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value in CSV string.")
    if any(v <= 0 for v in values):
        raise ValueError("All epsilon sweep values must be > 0.")
    return values


# Set random seeds for reproducible runs.
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Resolve whether to use CPU or CUDA device.
def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


# ==========================================
# 1. Model and Training Utilities
# ==========================================

# Build MNIST preprocessing: convert PIL image (pixel range 0...255) to tensor in [0, 1], then normalize
# using MNIST mean/std so inputs are centered and scaled for more stable training.
def get_mnist_transform() -> transforms.Compose:
    mean = (0.1307,)
    std = (0.3081,)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


# Construct the SimpleMLP model on the target device.
def build_model(device: torch.device) -> nn.Module:
    model = SimpleMLP()
    return model.to(device)


# Create DataLoaders for MNIST training and testing.
def create_data_loaders(
        data_dir: Path,
        download: bool,
        batch_size: int,
        num_workers: int,
        device: torch.device,
) -> Dict[str, DataLoader]:
    train_dataset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=True,
        transform=get_mnist_transform(),
        download=download,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=False,
        transform=get_mnist_transform(),
        download=download,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return {"train": train_loader, "test": test_loader}


# Train the model for one epoch and return loss/accuracy.
def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total += batch_size
        total_loss += loss.item() * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


# Evaluate model accuracy on a dataset without gradient tracking.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total += labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    accuracy = correct / max(total, 1)
    return {"accuracy": accuracy}


# ==================================================
# 2. Sensitivity Analysis and Saliency Map Utilities
# ==================================================

# Load one MNIST test sample by index.
def load_mnist_sample(
        data_dir: Path,
        index: int,
        download: bool,
) -> Tuple[torch.Tensor, int]:
    dataset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=False,
        transform=get_mnist_transform(),
        download=download,
    )
    if index < 0 or index >= len(dataset):
        raise IndexError(f"image-index {index} out of range [0, {len(dataset) - 1}]")
    image, label = dataset[index]
    return image, label


# Convert normalized image tensor back to [0, 1] for visualization.
def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.1307], device=image_tensor.device).view(1, 1, 1)
    std = torch.tensor([0.3081], device=image_tensor.device).view(1, 1, 1)
    image = image_tensor * std + mean
    image = torch.clamp(image, 0.0, 1.0)
    return image.detach().cpu().squeeze(0).numpy()


# Predict logits for a batched image tensor.
@torch.no_grad()
def predict_logits(model: nn.Module, image_bchw: torch.Tensor) -> torch.Tensor:
    return model(image_bchw)


# Compute input gradient for one class score.
def compute_input_gradient(
        model: nn.Module,
        image_bchw: torch.Tensor,
        class_index: int,
) -> torch.Tensor:
    image = image_bchw.clone().detach().requires_grad_(True)
    logits = model(image)
    score = logits[0, class_index]
    grad = torch.autograd.grad(score, image)[0]
    return grad.detach()


# Compute Jacobian using PyTorch jacobian API (full or selected classes).
def compute_jacobian(
        model: nn.Module,
        image_bchw: torch.Tensor,
        class_indices: Optional[List[int]] = None,
) -> torch.Tensor | Dict[int, torch.Tensor]:
    image_chw = image_bchw[0].clone().detach().requires_grad_(True)

    def logits_fn(input_chw: torch.Tensor) -> torch.Tensor:
        return model(input_chw.unsqueeze(0))[0]

    full_jacobian = torch.autograd.functional.jacobian(
        logits_fn,
        image_chw,
        create_graph=False,
        strict=False,
    ).detach()
    if class_indices is None:
        return full_jacobian
    return {class_idx: full_jacobian[class_idx].detach() for class_idx in class_indices}


# Compute and return full Jacobian tensor with explicit type.
def compute_jacobian_full(model: nn.Module, image_bchw: torch.Tensor) -> torch.Tensor:
    return compute_jacobian(model, image_bchw, class_indices=None)


# Compute and return selected-class Jacobian entries with explicit type.
def compute_jacobian_selected(
        model: nn.Module,
        image_bchw: torch.Tensor,
        class_indices: List[int],
) -> Dict[int, torch.Tensor]:
    return compute_jacobian(model, image_bchw, class_indices=class_indices)


# Reduce gradient tensor (BCHW or CHW) to a signed 2D saliency map.
def reduce_saliency(grad: torch.Tensor) -> np.ndarray:
    if grad.dim() == 4:
        grad_chw = grad[0]
    elif grad.dim() == 3:
        grad_chw = grad
    else:
        raise ValueError(f"Expected gradient with 3 or 4 dims, got shape {tuple(grad.shape)}")
    channel_idx = grad_chw.abs().argmax(dim=0, keepdim=True)
    reduced = grad_chw.gather(dim=0, index=channel_idx).squeeze(0)
    return reduced.detach().cpu().numpy()


# Normalize signed map values to [-1, 1] with symmetric percentile clipping.
def normalize_map(raw_map: np.ndarray) -> np.ndarray:
    max_abs = np.percentile(np.abs(raw_map), 99)
    denom = max(float(max_abs), 1e-8)
    clipped = np.clip(raw_map, -denom, denom)
    return clipped / denom


# Compute SmoothGrad saliency map.
def smoothgrad_saliency(
        model: nn.Module,
        image_bchw: torch.Tensor,
        class_index: int,
        samples: int,
        noise_std: float,
) -> np.ndarray:
    all_maps = []
    sigma = noise_std * (image_bchw.max() - image_bchw.min()).item()
    for _ in range(samples):
        noise = torch.randn_like(image_bchw) * sigma
        noisy = image_bchw + noise
        grad = compute_input_gradient(model, noisy, class_index)
        all_maps.append(reduce_saliency(grad))
    return normalize_map(np.mean(np.stack(all_maps, axis=0), axis=0))


# Compute Integrated Gradients saliency map.
def integrated_gradients_saliency(
        model: nn.Module,
        image_bchw: torch.Tensor,
        class_index: int,
        steps: int,
) -> np.ndarray:
    baseline = torch.zeros_like(image_bchw)
    total_grad = torch.zeros_like(image_bchw)

    for alpha in torch.linspace(0.0, 1.0, steps, device=image_bchw.device):
        interpolated = baseline + alpha * (image_bchw - baseline)
        grad = compute_input_gradient(model, interpolated, class_index)
        total_grad += grad

    avg_grad = total_grad / steps
    integrated_grad = (image_bchw - baseline) * avg_grad
    return normalize_map(reduce_saliency(integrated_grad))


# Compute Vanilla saliency maps for all classes from a single Jacobian call.
def vanilla_saliency_all_classes_from_jacobian(full_jacobian: torch.Tensor) -> Dict[int, np.ndarray]:
    num_classes = int(full_jacobian.shape[0])
    all_maps: Dict[int, np.ndarray] = {}
    for class_idx in range(num_classes):
        all_maps[class_idx] = normalize_map(reduce_saliency(full_jacobian[class_idx]))
    return all_maps


# Compute SmoothGrad saliency maps for all classes using Jacobian per noisy sample.
def smoothgrad_saliency_all_classes(
        model: nn.Module,
        image_bchw: torch.Tensor,
        samples: int,
        noise_std: float,
) -> Dict[int, np.ndarray]:
    if samples <= 0:
        raise ValueError("smoothgrad-samples must be > 0")

    sigma = noise_std * (image_bchw.max() - image_bchw.min()).item()
    all_class_maps: List[List[np.ndarray]] = []
    for _ in range(samples):
        noise = torch.randn_like(image_bchw) * sigma
        noisy = image_bchw + noise
        jacobian = compute_jacobian_full(model, noisy)
        sample_maps = [reduce_saliency(jacobian[class_idx]) for class_idx in range(jacobian.shape[0])]
        all_class_maps.append(sample_maps)

    mean_maps = np.mean(np.asarray(all_class_maps), axis=0)
    return {class_idx: normalize_map(mean_maps[class_idx]) for class_idx in range(mean_maps.shape[0])}


# Compute Integrated Gradients saliency maps for all classes using Jacobian per interpolation step.
def integrated_gradients_saliency_all_classes(
        model: nn.Module,
        image_bchw: torch.Tensor,
        steps: int,
) -> Dict[int, np.ndarray]:
    if steps <= 0:
        raise ValueError("ig-steps must be > 0")

    baseline = torch.zeros_like(image_bchw)
    accum_jacobian: Optional[torch.Tensor] = None

    for alpha in torch.linspace(0.0, 1.0, steps, device=image_bchw.device):
        interpolated = baseline + alpha * (image_bchw - baseline)
        jacobian = compute_jacobian_full(model, interpolated)
        if accum_jacobian is None:
            accum_jacobian = jacobian
        else:
            accum_jacobian = accum_jacobian + jacobian

    if accum_jacobian is None:
        raise RuntimeError("Failed to accumulate Integrated Gradients Jacobian.")

    avg_jacobian = accum_jacobian / steps
    delta = (image_bchw - baseline)[0]
    integrated_grad = avg_jacobian * delta
    return {
        class_idx: normalize_map(reduce_saliency(integrated_grad[class_idx]))
        for class_idx in range(integrated_grad.shape[0])
    }


# Save a side-by-side figure of input and saliency variants.
def save_saliency_figure(
        image: np.ndarray,
        vanilla_map: np.ndarray,
        smoothgrad_map: np.ndarray,
        ig_map: np.ndarray,
        title: str,
        output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    color_norm = Normalize(vmin=-1.0, vmax=1.0)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Input")
    # axes[1].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    im = axes[1].imshow(vanilla_map, cmap="seismic", norm=color_norm, alpha=1)
    axes[1].set_title("Vanilla Grad")
    # axes[2].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(smoothgrad_map, cmap="seismic", norm=color_norm, alpha=1)
    axes[2].set_title("SmoothGrad")
    # axes[3].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].imshow(ig_map, cmap="seismic", norm=color_norm, alpha=1)
    axes[3].set_title("Integrated Grad")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.82, bottom=0.05, wspace=0.15)
    cax = fig.add_axes([0.90, 0.14, 0.012, 0.62])

    colorbar = fig.colorbar(
        im,
        cax=cax,
        ticks=[-1.0, 0.0, 1.0],
    )
    colorbar.ax.set_ylabel("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# Save one figure containing all 10 class saliency maps for a single method.
def save_all_class_saliency_figure(
        method_name: str,
        all_class_maps: Dict[int, np.ndarray],
        pred_idx: int,
        true_label: int,
        output_path: Path,
) -> None:
    num_classes = len(MNIST_CLASSES)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    flat_axes = axes.flatten()
    color_norm = Normalize(vmin=-1.0, vmax=1.0)

    for class_idx in range(num_classes):
        ax = flat_axes[class_idx]
        im = ax.imshow(all_class_maps[class_idx], cmap="seismic", norm=color_norm)
        title = f"Class {MNIST_CLASSES[class_idx]}"
        if class_idx == pred_idx:
            title += " (Pred)"
            ax.set_title(title, color="tab:red")
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:red")
                spine.set_linewidth(2.0)
        else:
            ax.set_title(title)
        ax.axis("off")

    for idx in range(num_classes, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(
        f"{method_name} | Pred: {MNIST_CLASSES[pred_idx]} | True: {MNIST_CLASSES[true_label]}"
    )
    fig.subplots_adjust(left=0.02, right=0.90, top=0.88, bottom=0.05, wspace=0.15, hspace=0.3)
    cax = fig.add_axes([0.92, 0.14, 0.012, 0.72])

    colorbar = fig.colorbar(
        im,
        cax=cax,
        ticks=[-1.0, 0.0, 1.0],
    )
    colorbar.ax.set_ylabel("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# Load a trained checkpoint into a model for sensitivity analysis.
def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = build_model(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# Compute prediction details and Jacobian summary values.
def compute_prediction_and_jacobian_summary(
        model: nn.Module,
        image_bchw: torch.Tensor,
        topk_jacobian: int,
) -> Dict[str, Any]:
    logits = predict_logits(model, image_bchw)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    pred_prob = float(probs[0, pred_idx].item())

    topk = int(min(topk_jacobian, logits.shape[1]))
    topk_indices = torch.topk(logits[0], k=topk).indices.tolist()

    full_jacobian = compute_jacobian_full(model, image_bchw)
    jac = compute_jacobian_selected(model, image_bchw, topk_indices)
    jacobian_norms = {int(idx): float(jac[idx].norm().item()) for idx in topk_indices}

    return {
        "pred_idx": pred_idx,
        "pred_prob": pred_prob,
        "topk_indices": topk_indices,
        "full_jacobian": full_jacobian,
        "jacobian_norms": jacobian_norms,
    }


# Compute single-class and all-class saliency maps.
def compute_sensitivity_maps(
        model: nn.Module,
        image_bchw: torch.Tensor,
        pred_idx: int,
        full_jacobian: torch.Tensor,
        ig_steps: int,
        smoothgrad_samples: int,
        smoothgrad_noise_std: float,
) -> Dict[str, Any]:
    grad = compute_input_gradient(model, image_bchw, pred_idx)
    vanilla_map = normalize_map(reduce_saliency(grad))
    smooth_map = smoothgrad_saliency(
        model=model,
        image_bchw=image_bchw,
        class_index=pred_idx,
        samples=smoothgrad_samples,
        noise_std=smoothgrad_noise_std,
    )
    ig_map = integrated_gradients_saliency(
        model=model,
        image_bchw=image_bchw,
        class_index=pred_idx,
        steps=ig_steps,
    )

    all_class_vanilla_maps = vanilla_saliency_all_classes_from_jacobian(full_jacobian)
    all_class_smoothgrad_maps = smoothgrad_saliency_all_classes(
        model=model,
        image_bchw=image_bchw,
        samples=smoothgrad_samples,
        noise_std=smoothgrad_noise_std,
    )
    all_class_ig_maps = integrated_gradients_saliency_all_classes(
        model=model,
        image_bchw=image_bchw,
        steps=ig_steps,
    )

    return {
        "vanilla_map": vanilla_map,
        "smooth_map": smooth_map,
        "ig_map": ig_map,
        "all_class_vanilla_maps": all_class_vanilla_maps,
        "all_class_smoothgrad_maps": all_class_smoothgrad_maps,
        "all_class_ig_maps": all_class_ig_maps,
    }


# Save all sensitivity figure artifacts and return their paths.
def save_sensitivity_artifacts(
        output_dir: Path,
        image_index: int,
        image_np: np.ndarray,
        true_label: int,
        pred_idx: int,
        pred_prob: float,
        vanilla_map: np.ndarray,
        smooth_map: np.ndarray,
        ig_map: np.ndarray,
        all_class_vanilla_maps: Dict[int, np.ndarray],
        all_class_smoothgrad_maps: Dict[int, np.ndarray],
        all_class_ig_maps: Dict[int, np.ndarray],
) -> Dict[str, Path]:
    figure_path = output_dir / f"{OUTPUT_PREFIX}_saliency_idx_{image_index}.png"
    all_class_vanilla_path = output_dir / f"{OUTPUT_PREFIX}_saliency_all_vanilla_idx_{image_index}.png"
    all_class_smoothgrad_path = output_dir / f"{OUTPUT_PREFIX}_saliency_all_smoothgrad_idx_{image_index}.png"
    all_class_ig_path = output_dir / f"{OUTPUT_PREFIX}_saliency_all_integrated_gradients_idx_{image_index}.png"
    title = f"Pred: {MNIST_CLASSES[pred_idx]} ({pred_prob:.3f}) | True: {MNIST_CLASSES[true_label]}"

    save_saliency_figure(
        image=image_np,
        vanilla_map=vanilla_map,
        smoothgrad_map=smooth_map,
        ig_map=ig_map,
        title=title,
        output_path=figure_path,
    )
    save_all_class_saliency_figure(
        method_name="Vanilla Gradient (All Classes)",
        all_class_maps=all_class_vanilla_maps,
        pred_idx=pred_idx,
        true_label=true_label,
        output_path=all_class_vanilla_path,
    )
    save_all_class_saliency_figure(
        method_name="SmoothGrad (All Classes)",
        all_class_maps=all_class_smoothgrad_maps,
        pred_idx=pred_idx,
        true_label=true_label,
        output_path=all_class_smoothgrad_path,
    )
    save_all_class_saliency_figure(
        method_name="Integrated Gradients (All Classes)",
        all_class_maps=all_class_ig_maps,
        pred_idx=pred_idx,
        true_label=true_label,
        output_path=all_class_ig_path,
    )

    return {
        "saliency_figure": figure_path,
        "saliency_all_classes_vanilla": all_class_vanilla_path,
        "saliency_all_classes_smoothgrad": all_class_smoothgrad_path,
        "saliency_all_classes_integrated_gradients": all_class_ig_path,
    }


# Write sensitivity section into the report JSON.
def write_sensitivity_report(
        report_path: Path,
        checkpoint_path: Path,
        args: argparse.Namespace,
        device: torch.device,
        true_label: int,
        pred_idx: int,
        pred_prob: float,
        topk_indices: List[int],
        jacobian_norms: Dict[int, float],
        artifact_paths: Dict[str, Path],
) -> None:
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {
            "model": {
                "source": "local-train",
                "name": "SimpleMLP",
                "checkpoint_path": str(checkpoint_path),
            },
        }

    report["sensitivity"] = {
        "seed": args.seed,
        "device": str(device),
        "image_index": args.image_index,
        "true_label": {"index": true_label, "name": MNIST_CLASSES[true_label]},
        "prediction": {
            "index": pred_idx,
            "name": MNIST_CLASSES[pred_idx],
            "probability": pred_prob,
        },
        "jacobian_class_indices": topk_indices,
        "jacobian_l2_norms": {MNIST_CLASSES[idx]: jacobian_norms[idx] for idx in topk_indices},
        "artifacts": {
            "saliency_figure": str(artifact_paths["saliency_figure"]),
            "report": str(report_path),
            "saliency_all_classes_vanilla": str(artifact_paths["saliency_all_classes_vanilla"]),
            "saliency_all_classes_smoothgrad": str(artifact_paths["saliency_all_classes_smoothgrad"]),
            "saliency_all_classes_integrated_gradients": str(
                artifact_paths["saliency_all_classes_integrated_gradients"]),
        },
        "settings": {
            "topk_jacobian": args.topk_jacobian,
            "ig_steps": args.ig_steps,
            "smoothgrad_samples": args.smoothgrad_samples,
            "smoothgrad_noise_std": args.smoothgrad_noise_std,
        },
        "all_class_map_order": list(range(len(MNIST_CLASSES))),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# Run the full training, checkpointing, and evaluation pipeline.
def main_train(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / CHECKPOINT_FILENAME

    loaders = create_data_loaders(
        data_dir=args.data_dir,
        download=args.download_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = build_model(device=device)
    history = []
    loaded_from_checkpoint = False
    if checkpoint_path.exists():
        if args.force_train:
            print(f"Checkpoint exists at {checkpoint_path} but --force-train is enabled. Retraining model.")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                history = checkpoint.get("history", [])
            else:
                model.load_state_dict(checkpoint)
            loaded_from_checkpoint = True
            print(f"Loaded existing model from {checkpoint_path}. Skipping training.")

    if not loaded_from_checkpoint:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            test_stats = evaluate(model=model, loader=loaders["test"], device=device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_stats["loss"]),
                    "train_accuracy": float(train_stats["accuracy"]),
                    "test_accuracy": float(test_stats["accuracy"]),
                }
            )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_stats['loss']:.4f} "
                f"train_acc={train_stats['accuracy']:.4f} "
                f"test_acc={test_stats['accuracy']:.4f}"
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )
        print("Saved trained model:", checkpoint_path)

    final_test = evaluate(model=model, loader=loaders["test"], device=device)["accuracy"]

    report_path = args.output_dir / REPORT_FILENAME
    report = {
        "model": {
            "source": "local-train",
            "name": "SimpleMLP",
            "checkpoint_path": str(checkpoint_path),
            "loaded_from_checkpoint": loaded_from_checkpoint,
        },
        "training": {
            "seed": args.seed,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        "final_test_accuracy": float(final_test),
        "history": history,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved report:", report_path)
    print(f"Final MNIST test accuracy: {final_test:.4f}")


# Run sensitivity analysis and saliency map generation using a trained checkpoint.
def main_sensitivity(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run training first (--run-train)."
        )

    model = load_checkpoint_model(checkpoint_path=checkpoint_path, device=device)

    image_chw, true_label = load_mnist_sample(
        data_dir=args.data_dir,
        index=args.image_index,
        download=args.download_dataset,
    )
    image_bchw = image_chw.unsqueeze(0).to(device)

    prediction_data = compute_prediction_and_jacobian_summary(
        model=model,
        image_bchw=image_bchw,
        topk_jacobian=args.topk_jacobian,
    )
    pred_idx = prediction_data["pred_idx"]
    pred_prob = prediction_data["pred_prob"]
    topk_indices = prediction_data["topk_indices"]
    full_jacobian = prediction_data["full_jacobian"]
    jacobian_norms = prediction_data["jacobian_norms"]

    map_data = compute_sensitivity_maps(
        model=model,
        image_bchw=image_bchw,
        pred_idx=pred_idx,
        full_jacobian=full_jacobian,
        ig_steps=args.ig_steps,
        smoothgrad_samples=args.smoothgrad_samples,
        smoothgrad_noise_std=args.smoothgrad_noise_std,
    )

    artifact_paths = save_sensitivity_artifacts(
        output_dir=args.output_dir,
        image_index=args.image_index,
        image_np=denormalize(image_bchw[0]),
        true_label=true_label,
        pred_idx=pred_idx,
        pred_prob=pred_prob,
        vanilla_map=map_data["vanilla_map"],
        smooth_map=map_data["smooth_map"],
        ig_map=map_data["ig_map"],
        all_class_vanilla_maps=map_data["all_class_vanilla_maps"],
        all_class_smoothgrad_maps=map_data["all_class_smoothgrad_maps"],
        all_class_ig_maps=map_data["all_class_ig_maps"],
    )

    report_path = args.output_dir / REPORT_FILENAME
    write_sensitivity_report(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        args=args,
        device=device,
        true_label=true_label,
        pred_idx=pred_idx,
        pred_prob=pred_prob,
        topk_indices=topk_indices,
        jacobian_norms=jacobian_norms,
        artifact_paths=artifact_paths,
    )

    print("Loaded checkpoint:", checkpoint_path)
    print("True label:", MNIST_CLASSES[true_label])
    print("Prediction:", MNIST_CLASSES[pred_idx], f"(p={pred_prob:.3f})")
    print("Top-k class indices for Jacobian:", topk_indices)
    print("Saved figure:", artifact_paths["saliency_figure"])
    print("Saved all-class vanilla figure:", artifact_paths["saliency_all_classes_vanilla"])
    print("Saved all-class smoothgrad figure:", artifact_paths["saliency_all_classes_smoothgrad"])
    print("Saved all-class integrated-gradients figure:", artifact_paths["saliency_all_classes_integrated_gradients"])
    print("Updated report:", report_path)


def sample_l2_perturbation(reference: torch.Tensor, eps: float) -> torch.Tensor:
    noise = torch.randn_like(reference)
    flat = noise.view(noise.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    unit = flat / norms
    return (unit * eps).view_as(reference)


def flatten_jacobian(full_jacobian: torch.Tensor) -> torch.Tensor:
    return full_jacobian.reshape(full_jacobian.shape[0], -1)


def local_lipschitz_from_jacobian(full_jacobian: torch.Tensor) -> float:
    jac_flat = flatten_jacobian(full_jacobian)
    singular_values = torch.linalg.svdvals(jac_flat)
    return float(singular_values.max().item())


@torch.no_grad()
def estimate_margin(logits: torch.Tensor) -> Tuple[int, float, int]:
    pred_idx = int(torch.argmax(logits, dim=1).item())
    values, indices = torch.topk(logits[0], k=2)
    runner_up_idx = int(indices[1].item())
    margin = float((values[0] - values[1]).item())
    return pred_idx, margin, runner_up_idx


# Section 3.3: compute Jacobian spectral norm as a local Lipschitz sensitivity indicator.
def section_3_3_spectral_norm_of_jacobian(full_jacobian: torch.Tensor) -> Dict[str, Any]:
    local_lipschitz = local_lipschitz_from_jacobian(full_jacobian)
    return {
        "jacobian_shape": list(full_jacobian.shape),
        "local_lipschitz_spectral_norm": local_lipschitz,
    }


def section_3_2_local_linearization_validation(
        model: nn.Module,
        image_bchw: torch.Tensor,
        base_logits: torch.Tensor,
        jac_flat: torch.Tensor,
        eps: float,
        trials: int,
        local_lipschitz: float,
) -> Dict[str, Any]:
    trial_actual_norms: List[float] = []
    trial_linear_norms: List[float] = []
    trial_linearization_errors: List[float] = []
    bound_value = local_lipschitz * eps
    bound_violations = 0

    # Compare true output shift with first-order linearized shift under controlled L2 perturbations.
    for _ in range(trials):
        delta = sample_l2_perturbation(image_bchw, eps)
        perturbed_logits = predict_logits(model, image_bchw + delta)

        delta_y_actual = (perturbed_logits - base_logits)[0]
        delta_x_flat = delta[0].reshape(-1)
        delta_y_linear = jac_flat @ delta_x_flat

        actual_norm = float(torch.norm(delta_y_actual, p=2).item())
        linear_norm = float(torch.norm(delta_y_linear, p=2).item())
        lin_error = float(torch.norm(delta_y_actual - delta_y_linear, p=2).item())

        trial_actual_norms.append(actual_norm)
        trial_linear_norms.append(linear_norm)
        trial_linearization_errors.append(lin_error)
        if actual_norm > bound_value + 1e-8:
            bound_violations += 1

    return {
        "trial_actual_norms": trial_actual_norms,
        "trial_linear_norms": trial_linear_norms,
        "trial_linearization_errors": trial_linearization_errors,
        "bound_lipschitz_eps": bound_value,
        "bound_violations": bound_violations,
        "bound_violation_rate": bound_violations / max(trials, 1),
        "actual_output_shift_l2_mean": float(np.mean(trial_actual_norms)),
        "linearized_output_shift_l2_mean": float(np.mean(trial_linear_norms)),
        "linearization_error_l2_mean": float(np.mean(trial_linearization_errors)),
    }


# Section 3.4: estimate robustness radius lower bound from margin and local Lipschitz.
def section_3_4_robustness_radius_estimation(margin: float, local_lipschitz: float) -> Dict[str, float]:
    radius_lower = margin / (np.sqrt(2.0) * max(local_lipschitz, 1e-12))
    return {
        "margin_logit": margin,
        "robustness_radius_lower_bound": radius_lower,
    }


# Section 3.5: empirically validate robustness via label-flip rate under random perturbations.
def section_3_5_empirical_validation(
        model: nn.Module,
        data_dir: Path,
        download_dataset: bool,
        eps: float,
        eval_samples: int,
        device: torch.device,
) -> Dict[str, Any]:
    test_dataset = torchvision.datasets.MNIST(
        root=str(data_dir),
        train=False,
        transform=get_mnist_transform(),
        download=download_dataset,
    )

    empirical_flip_count = 0
    evaluated_samples = min(eval_samples, len(test_dataset))
    for i in range(evaluated_samples):
        sample_img, _ = test_dataset[i]
        sample_bchw = sample_img.unsqueeze(0).to(device)
        base_pred = int(torch.argmax(predict_logits(model, sample_bchw), dim=1).item())
        delta = sample_l2_perturbation(sample_bchw, eps)
        perturbed_pred = int(torch.argmax(predict_logits(model, sample_bchw + delta), dim=1).item())
        if perturbed_pred != base_pred:
            empirical_flip_count += 1

    flip_rate = empirical_flip_count / max(evaluated_samples, 1)
    return {
        "evaluated_samples": evaluated_samples,
        "label_flip_count": empirical_flip_count,
        "label_flip_rate": flip_rate,
    }


def save_robustness_linearization_figure(
        trial_actual_norms: List[float],
        trial_linear_norms: List[float],
        bound_value: float,
        output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(1, len(trial_actual_norms) + 1)

    axes[0].plot(x, trial_actual_norms, label="Actual ||delta y||2", color="tab:blue")
    axes[0].plot(x, trial_linear_norms, label="Linearized ||J delta x||2", color="tab:orange")
    axes[0].axhline(bound_value, color="tab:red", linestyle="--", label="L(x) * eps")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Output shift norm")
    axes[0].set_title("Local Linearization Validation")
    axes[0].legend()

    means = [
        float(np.mean(trial_actual_norms)),
        float(np.mean(trial_linear_norms)),
        bound_value,
    ]
    axes[1].bar(["Actual Mean", "Linear Mean", "Bound"], means, color=["tab:blue", "tab:orange", "tab:red"])
    axes[1].set_ylabel("Value")
    axes[1].set_title("Summary")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# Build the section 3.5 result figure: flip counts and corresponding flip/non-flip rates.
def save_empirical_validation_figure(
        evaluated_samples: int,
        label_flip_count: int,
        output_path: Path,
) -> None:
    non_flip_count = max(evaluated_samples - label_flip_count, 0)
    flip_rate = label_flip_count / max(evaluated_samples, 1)
    non_flip_rate = 1.0 - flip_rate

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    count_bars = axes[0].bar(
        ["Not Flipped", "Flipped"],
        [non_flip_count, label_flip_count],
        color=["tab:green", "tab:red"],
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Flip Counts")
    for bar, value in zip(count_bars, [non_flip_count, label_flip_count]):
        height = bar.get_height()
        y = height + 0.05 * max(evaluated_samples, 1)
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, y, f"{value}", ha="center", va="bottom")

    rate_bars = axes[1].bar(
        ["Not Flipped Rate", "Flip Rate"],
        [non_flip_rate, flip_rate],
        color=["tab:green", "tab:blue"],
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Empirical Flip Rate")
    for bar, value in zip(rate_bars, [non_flip_rate, flip_rate]):
        height = bar.get_height()
        y = height + 0.02
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, y, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_spectral_norm_figure(
        local_lipschitz: float,
        output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars = ax.bar(["Local Spectral Norm"], [local_lipschitz], color=["tab:purple"])
    ax.set_ylabel("Value")
    ax.set_title("Section 3.3: Jacobian Spectral Norm")
    bar = bars[0]
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02 * max(height, 1.0), f"{local_lipschitz:.4f}",
            ha="center", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_radius_estimation_figure(
        margin_logit: float,
        local_lipschitz: float,
        radius_lower: float,
        eps: float,
        output_path: Path,
) -> None:
    denom_term = np.sqrt(2.0) * local_lipschitz
    bound_term = local_lipschitz * eps

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bars_left = axes[0].bar(
        ["Margin", "sqrt(2)*L(x)"],
        [margin_logit, denom_term],
        color=["tab:blue", "tab:orange"],
    )
    axes[0].set_ylabel("Value")
    axes[0].set_title("Section 3.4: Radius Terms")
    for bar, value in zip(bars_left, [margin_logit, denom_term]):
        height = bar.get_height()
        y = height + 0.02 * max(height, 1.0)
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, y, f"{value:.3f}", ha="center", va="bottom")

    bars_right = axes[1].bar(
        ["Radius Lower Bound", "L(x)*eps"],
        [radius_lower, bound_term],
        color=["tab:green", "tab:red"],
    )
    axes[1].set_ylabel("Value")
    axes[1].set_title("Estimated Radius vs Local Bound")
    for bar, value in zip(bars_right, [radius_lower, bound_term]):
        height = bar.get_height()
        y = height + 0.02 * max(height, 1.0)
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, y, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_eps_sweep_figure(
        eps_values: List[float],
        flip_rates: List[float],
        bound_violation_rates: List[float],
        output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eps_values, flip_rates, marker="o", label="Empirical Flip Rate", color="tab:blue", linewidth=2)
    ax.plot(eps_values, bound_violation_rates, marker="s", label="Bound Violation Rate", color="tab:red", linewidth=2)
    for x, y in zip(eps_values, flip_rates):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", color="tab:blue",
                    fontsize=8)
    for x, y in zip(eps_values, bound_violation_rates):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", color="tab:red",
                    fontsize=8)
    if all(v == 0.0 for v in flip_rates) and all(v == 0.0 for v in bound_violation_rates):
        ax.text(0.5, 0.08, "All observed rates are zero at tested eps values", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9)
    ax.set_xlabel("Epsilon (L2 perturbation budget)")
    ax.set_ylabel("Rate")
    ax.set_title("Epsilon Sweep: Empirical vs Analytical Indicators")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_spectral_norm_distribution_figure(
        local_lipschitz_values: List[float],
        output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(local_lipschitz_values, bins=min(10, max(3, len(local_lipschitz_values))), color="tab:purple", alpha=0.8,
            edgecolor="white")
    ax.set_xlabel("Local Spectral Norm")
    ax.set_ylabel("Sample Count")
    ax.set_title("Distribution of Local Jacobian Spectral Norms")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_radius_vs_flip_scatter_figure(
        radius_values: List[float],
        flip_rates: List[float],
        output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    all_zero = all(v == 0.0 for v in flip_rates)
    if all_zero:
        y_vis = [0.02 for _ in flip_rates]
        ax.scatter(radius_values, y_vis, color="tab:green", alpha=0.9, marker="x", s=55)
        ax.text(0.5, 0.08, "All true per-sample flip rates are zero", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9)
    else:
        ax.scatter(radius_values, flip_rates, color="tab:green", alpha=0.85)
    ax.set_xlabel("Robustness Radius Lower Bound")
    ax.set_ylabel("Per-sample Flip Rate")
    ax.set_title("Analytical vs Empirical Robustness")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_robustness_report(
        report_path: Path,
        checkpoint_path: Path,
        args: argparse.Namespace,
        device: torch.device,
        robustness_payload: Dict[str, Any],
) -> None:
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {
            "model": {
                "source": "local-train",
                "name": "SimpleMLP",
                "checkpoint_path": str(checkpoint_path),
            },
        }

    report["robustness"] = {
        "seed": args.seed,
        "device": str(device),
        "settings": {
            "image_index": args.image_index,
            "eps": args.robustness_eps,
            "trials": args.robustness_trials,
            "eval_samples": args.robustness_eval_samples,
        },
        **robustness_payload,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main_robustness(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run training first (--run-train)."
        )

    model = load_checkpoint_model(checkpoint_path=checkpoint_path, device=device)
    image_chw, true_label = load_mnist_sample(
        data_dir=args.data_dir,
        index=args.image_index,
        download=args.download_dataset,
    )
    image_bchw = image_chw.unsqueeze(0).to(device)

    base_logits = predict_logits(model, image_bchw)
    pred_idx, margin, runner_up_idx = estimate_margin(base_logits)
    full_jacobian = compute_jacobian_full(model, image_bchw)
    jac_flat = flatten_jacobian(full_jacobian)

    # Section 3.3 execution in the robustness pipeline.
    section_3_3 = section_3_3_spectral_norm_of_jacobian(full_jacobian)
    local_lipschitz = float(section_3_3["local_lipschitz_spectral_norm"])
    section_3_2 = section_3_2_local_linearization_validation(
        model=model,
        image_bchw=image_bchw,
        base_logits=base_logits,
        jac_flat=jac_flat,
        eps=args.robustness_eps,
        trials=args.robustness_trials,
        local_lipschitz=local_lipschitz,
    )
    # Section 3.4 execution: compute radius estimate using margin and spectral norm.
    section_3_4 = section_3_4_robustness_radius_estimation(margin=margin, local_lipschitz=local_lipschitz)
    # Section 3.5 execution: estimate empirical flip rate on perturbed MNIST samples.
    section_3_5 = section_3_5_empirical_validation(
        # Reuse the same perturbation budget for a direct analytical-vs-empirical comparison.
        model=model,
        data_dir=args.data_dir,
        download_dataset=args.download_dataset,
        eps=args.robustness_eps,
        eval_samples=args.robustness_eval_samples,
        device=device,
    )

    sweep_eps_values = parse_float_csv(args.robustness_sweep_eps)
    sweep_flip_rates: List[float] = []
    sweep_bound_violation_rates: List[float] = []
    for sweep_eps in sweep_eps_values:
        sweep_sec32 = section_3_2_local_linearization_validation(
            model=model,
            image_bchw=image_bchw,
            base_logits=base_logits,
            jac_flat=jac_flat,
            eps=sweep_eps,
            trials=args.robustness_trials,
            local_lipschitz=local_lipschitz,
        )
        sweep_sec35 = section_3_5_empirical_validation(
            model=model,
            data_dir=args.data_dir,
            download_dataset=args.download_dataset,
            eps=sweep_eps,
            eval_samples=args.robustness_eval_samples,
            device=device,
        )
        sweep_flip_rates.append(float(sweep_sec35["label_flip_rate"]))
        sweep_bound_violation_rates.append(float(sweep_sec32["bound_violation_rate"]))

    test_dataset = torchvision.datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        transform=get_mnist_transform(),
        download=args.download_dataset,
    )
    dist_sample_count = min(args.robustness_dist_samples, len(test_dataset))
    local_lipschitz_values: List[float] = []
    radius_values: List[float] = []
    per_sample_flip_rates: List[float] = []
    for i in range(dist_sample_count):
        sample_img, _ = test_dataset[i]
        sample_bchw = sample_img.unsqueeze(0).to(device)
        sample_logits = predict_logits(model, sample_bchw)
        base_pred, sample_margin, _ = estimate_margin(sample_logits)
        sample_jacobian = compute_jacobian_full(model, sample_bchw)
        sample_lipschitz = local_lipschitz_from_jacobian(sample_jacobian)
        sample_radius = section_3_4_robustness_radius_estimation(sample_margin, sample_lipschitz)[
            "robustness_radius_lower_bound"]
        local_lipschitz_values.append(float(sample_lipschitz))
        radius_values.append(float(sample_radius))

        flip_count = 0
        for _ in range(args.robustness_scatter_trials):
            delta = sample_l2_perturbation(sample_bchw, args.robustness_eps)
            perturbed_pred = int(torch.argmax(predict_logits(model, sample_bchw + delta), dim=1).item())
            if perturbed_pred != base_pred:
                flip_count += 1
        per_sample_flip_rates.append(flip_count / max(args.robustness_scatter_trials, 1))

    figure_path = args.output_dir / f"{OUTPUT_PREFIX}_robustness_linearization_idx_{args.image_index}.png"
    empirical_figure_path = args.output_dir / f"{OUTPUT_PREFIX}_empirical_validation_idx_{args.image_index}.png"
    spectral_figure_path = args.output_dir / f"{OUTPUT_PREFIX}_spectral_norm_idx_{args.image_index}.png"
    radius_figure_path = args.output_dir / f"{OUTPUT_PREFIX}_radius_estimation_idx_{args.image_index}.png"
    eps_sweep_figure_path = args.output_dir / f"{OUTPUT_PREFIX}_eps_sweep_idx_{args.image_index}.png"
    spectral_dist_figure_path = args.output_dir / f"{OUTPUT_PREFIX}_spectral_norm_distribution_idx_{args.image_index}.png"
    radius_scatter_figure_path = args.output_dir / f"{OUTPUT_PREFIX}_radius_vs_flip_scatter_idx_{args.image_index}.png"
    save_robustness_linearization_figure(
        trial_actual_norms=section_3_2["trial_actual_norms"],
        trial_linear_norms=section_3_2["trial_linear_norms"],
        bound_value=float(section_3_2["bound_lipschitz_eps"]),
        output_path=figure_path,
    )
    save_spectral_norm_figure(
        local_lipschitz=local_lipschitz,
        output_path=spectral_figure_path,
    )
    save_radius_estimation_figure(
        margin_logit=float(section_3_4["margin_logit"]),
        local_lipschitz=local_lipschitz,
        radius_lower=float(section_3_4["robustness_radius_lower_bound"]),
        eps=args.robustness_eps,
        output_path=radius_figure_path,
    )
    save_empirical_validation_figure(
        evaluated_samples=int(section_3_5["evaluated_samples"]),
        label_flip_count=int(section_3_5["label_flip_count"]),
        output_path=empirical_figure_path,
    )
    save_eps_sweep_figure(
        eps_values=sweep_eps_values,
        flip_rates=sweep_flip_rates,
        bound_violation_rates=sweep_bound_violation_rates,
        output_path=eps_sweep_figure_path,
    )
    save_spectral_norm_distribution_figure(
        local_lipschitz_values=local_lipschitz_values,
        output_path=spectral_dist_figure_path,
    )
    save_radius_vs_flip_scatter_figure(
        radius_values=radius_values,
        flip_rates=per_sample_flip_rates,
        output_path=radius_scatter_figure_path,
    )

    robustness_payload: Dict[str, Any] = {
        "true_label": {"index": true_label, "name": MNIST_CLASSES[true_label]},
        "prediction": {
            "index": pred_idx,
            "name": MNIST_CLASSES[pred_idx],
            "runner_up_index": runner_up_idx,
            "runner_up_name": MNIST_CLASSES[runner_up_idx],
        },
        "jacobian": {
            "shape": list(full_jacobian.shape),
            "local_lipschitz_spectral_norm": local_lipschitz,
        },
        # Section 3.4 results stored for report writing.
        "robustness_radius_lower_bound": section_3_4["robustness_radius_lower_bound"],
        "margin_logit": section_3_4["margin_logit"],
        "linearization_validation": {
            "actual_output_shift_l2_mean": section_3_2["actual_output_shift_l2_mean"],
            "linearized_output_shift_l2_mean": section_3_2["linearized_output_shift_l2_mean"],
            "linearization_error_l2_mean": section_3_2["linearization_error_l2_mean"],
            "bound_lipschitz_eps": section_3_2["bound_lipschitz_eps"],
            "bound_violations": section_3_2["bound_violations"],
            "bound_violation_rate": section_3_2["bound_violation_rate"],
        },
        # Section 3.5 results stored for report writing.
        "empirical_validation": section_3_5,
        "section_3_2_local_linearization": {
            "eps": args.robustness_eps,
            "trials": args.robustness_trials,
            "result_summary": {
                "actual_mean": section_3_2["actual_output_shift_l2_mean"],
                "linearized_mean": section_3_2["linearized_output_shift_l2_mean"],
                "linearization_error_mean": section_3_2["linearization_error_l2_mean"],
            },
        },
        "section_3_3_spectral_norm": section_3_3,
        "section_3_4_radius_estimation": section_3_4,
        "section_3_5_empirical_validation": section_3_5,
        "additional_analyses": {
            "epsilon_sweep": {
                "eps_values": sweep_eps_values,
                "flip_rates": sweep_flip_rates,
                "bound_violation_rates": sweep_bound_violation_rates,
            },
            "spectral_norm_distribution": {
                "sample_count": dist_sample_count,
                "mean": float(np.mean(local_lipschitz_values)) if local_lipschitz_values else 0.0,
                "std": float(np.std(local_lipschitz_values)) if local_lipschitz_values else 0.0,
            },
            "radius_vs_flip_scatter": {
                "sample_count": dist_sample_count,
                "scatter_trials": args.robustness_scatter_trials,
            },
        },
        "artifacts": {
            "linearization_figure": str(figure_path),
            "spectral_norm_figure": str(spectral_figure_path),
            "radius_estimation_figure": str(radius_figure_path),
            "empirical_validation_figure": str(empirical_figure_path),
            "epsilon_sweep_figure": str(eps_sweep_figure_path),
            "spectral_norm_distribution_figure": str(spectral_dist_figure_path),
            "radius_vs_flip_scatter_figure": str(radius_scatter_figure_path),
        },
    }

    report_path = args.output_dir / REPORT_FILENAME
    write_robustness_report(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        args=args,
        device=device,
        robustness_payload=robustness_payload,
    )

    print("Loaded checkpoint:", checkpoint_path)
    print("Robustness true label:", MNIST_CLASSES[true_label])
    print("Robustness prediction:", MNIST_CLASSES[pred_idx])
    print(f"Local Lipschitz (spectral norm): {local_lipschitz:.6f}")
    print(f"Logit margin (pred-runner-up): {section_3_4['margin_logit']:.6f}")
    print(f"Robust radius lower bound: {section_3_4['robustness_radius_lower_bound']:.6f}")
    print(f"Random perturbation flip rate @ eps={args.robustness_eps:.4f}: {section_3_5['label_flip_rate']:.4f}")
    print("Saved robustness figure:", figure_path)
    print("Saved spectral-norm figure:", spectral_figure_path)
    print("Saved radius-estimation figure:", radius_figure_path)
    print("Saved empirical validation figure:", empirical_figure_path)
    print("Saved epsilon-sweep figure:", eps_sweep_figure_path)
    print("Saved spectral-norm distribution figure:", spectral_dist_figure_path)
    print("Saved radius-vs-flip scatter figure:", radius_scatter_figure_path)
    print("Updated report:", report_path)


def main_adversarial(args: Optional[argparse.Namespace] = None) -> None:
    """
    Investigates a holistic suite of adversarial perturbation methods:
    Baseline: FGSM, PGD (Linf), DeepFool, C&W (L2)
    Skews: PGD (L2), PGD (Targeted), C&W (High Confidence)
    """
    if args is None:
        args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run training first.")

    model = load_checkpoint_model(checkpoint_path, device)
    criterion = nn.CrossEntropyLoss()

    # --- Baseline Attack Implementations ---

    def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return torch.clamp(perturbed_image, -0.4242, 2.8215)

    def pgd_linf_attack(model, images, labels, eps, alpha, steps):
        adv_images = images.clone().detach().requires_grad_(True)
        for _ in range(steps):
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                adv_images = adv_images + alpha * adv_images.grad.sign()
                eta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + eta, -0.4242, 2.8215)
            adv_images.requires_grad = True
        return adv_images.detach()

    def deepfool_attack(model, image, num_classes=10, max_iter=50):
        image = image.clone().detach().requires_grad_(True)
        f_x = model(image).data.flatten()
        I = f_x.argsort(descending=True)
        label = I[0]

        input_shape = image.shape
        x_i = image.clone().detach().requires_grad_(True)

        for _ in range(max_iter):
            output = model(x_i)
            if output.data.argmax().item() != label:
                break
            pert = np.inf
            w = torch.zeros(input_shape).to(device)
            model.zero_grad()
            output[0, label].backward(retain_graph=True)
            grad_orig = x_i.grad.data.clone()
            for k in range(1, num_classes):
                x_i.grad.zero_()
                output[0, I[k]].backward(retain_graph=True)
                cur_grad = x_i.grad.data.clone()
                w_k = cur_grad - grad_orig
                f_k = (output[0, I[k]] - output[0, label]).data
                pert_k = torch.abs(f_k) / (torch.norm(w_k.flatten()) + 1e-8)
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = (pert + 1e-4) * w / (torch.norm(w.flatten()) + 1e-8)
            x_i.data = x_i.data + r_i
            x_i.grad.zero_()
        return torch.clamp(x_i, -0.4242, 2.8215).detach()

    def cw_attack(model, image, target_label, c=1.0, lr=0.01, steps=50):
        delta = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            adv_img = torch.clamp(image + delta, -0.4242, 2.8215)
            outputs = model(adv_img)
            one_hot_target = torch.eye(10, device=device)[target_label]
            target_logit = (outputs * one_hot_target).sum(1)
            other_logit = (outputs * (1 - one_hot_target) - one_hot_target * 10000).max(1)[0]
            loss = torch.norm(delta, p=2) + c * torch.clamp(other_logit - target_logit, min=0.0).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return torch.clamp(image + delta, -0.4242, 2.8215).detach()

    # --- Skewed Attack Implementations ---

    def pgd_l2_attack(model, images, labels, eps, alpha, steps):
        """PGD with L2-norm constraint (Constraint Skew)."""
        adv_images = images.clone().detach().requires_grad_(True)
        for _ in range(steps):
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = adv_images.grad
                grad_norms = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                grad_normalized = grad / (grad_norms + 1e-8)
                adv_images = adv_images + alpha * grad_normalized

                # Project back to L2 ball
                eta = adv_images - images
                eta_norms = torch.norm(eta.view(eta.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                factor = eps / (eta_norms + 1e-8)
                factor = torch.clamp(factor, max=1.0)
                adv_images = images + eta * factor
                adv_images = torch.clamp(adv_images, -0.4242, 2.8215)
            adv_images.requires_grad = True
        return adv_images.detach()

    def pgd_targeted_attack(model, images, target_labels, eps, alpha, steps):
        """Targeted PGD (Targeted Attack Skew). Minimizes loss for target class."""
        adv_images = images.clone().detach().requires_grad_(True)
        for _ in range(steps):
            outputs = model(adv_images)
            # Minimize the loss for the target class
            loss = criterion(outputs, target_labels)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                # Move in direction that DECREASES target loss
                adv_images = adv_images - alpha * adv_images.grad.sign()
                eta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + eta, -0.4242, 2.8215)
            adv_images.requires_grad = True
        return adv_images.detach()

    def cw_high_confidence_attack(model, image, target_label, kappa=20.0, c=1.0, lr=0.01, steps=100):
        """C&W Skew: High-Confidence optimization using kappa parameter."""
        delta = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            adv_img = torch.clamp(image + delta, -0.4242, 2.8215)
            outputs = model(adv_img)
            one_hot_target = torch.eye(10, device=device)[target_label]
            target_logit = (outputs * one_hot_target).sum(1)
            other_logit = (outputs - one_hot_target * 10000).max(1)[0]

            # Force high confidence misclassification
            loss_f = torch.clamp(other_logit - target_logit + kappa, min=0.0).sum()
            loss = torch.norm(delta, p=2) + c * loss_f
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return torch.clamp(image + delta, -0.4242, 2.8215).detach()

    # --- Holistic Evaluation Suite ---

    loaders = create_data_loaders(args.data_dir, args.download_dataset, args.batch_size, args.num_workers, device)
    limit = args.robustness_eval_samples

    methods = ["Clean", "FGSM", "PGD-Linf", "DeepFool", "CW-L2", "PGD-L2", "PGD-Targeted", "CW-HighConf"]
    results = {m: {"correct": 0, "targeted_success": 0, "total": 0, "examples": []} for m in methods}

    print(f"Investigating holistic adversarial suite on {limit} samples (eps={args.robustness_eps})...")

    for images, labels in loaders["test"]:
        if results["Clean"]["total"] >= limit: break
        images, labels = images.to(device), labels.to(device)

        # Targets for targeted attacks (runner-up)
        logits = model(images)
        _, top_classes = torch.topk(logits, k=2, dim=1)
        runner_up_labels = top_classes[:, 1]

        # Generate Batch Attacks
        images.requires_grad = True
        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        adv_fgsm = fgsm_attack(images, args.robustness_eps, data_grad)
        adv_pgd_linf = pgd_linf_attack(model, images, labels, args.robustness_eps, 0.01, 40)
        adv_pgd_l2 = pgd_l2_attack(model, images, labels, eps=2.0, alpha=0.1, steps=40)
        adv_pgd_targeted = pgd_targeted_attack(model, images, runner_up_labels, args.robustness_eps, 0.01, 40)

        for i in range(len(labels)):
            if results["Clean"]["total"] >= limit: break
            results["Clean"]["total"] += 1

            # Single sample attacks (expensive algorithms)
            single_img = images[i:i + 1]
            single_label = labels[i:i + 1]
            single_target = runner_up_labels[i:i + 1]

            adv_df = deepfool_attack(model, single_img)
            target_cw = (single_label.item() + 1) % 10
            adv_cw = cw_attack(model, single_img, target_cw)
            adv_cw_highconf = cw_high_confidence_attack(model, single_img, single_target.item(), kappa=20.0)

            batch_adv = {
                "Clean": single_img,
                "FGSM": adv_fgsm[i:i + 1],
                "PGD-Linf": adv_pgd_linf[i:i + 1],
                "DeepFool": adv_df,
                "CW-L2": adv_cw,
                "PGD-L2": adv_pgd_l2[i:i + 1],
                "PGD-Targeted": adv_pgd_targeted[i:i + 1],
                "CW-HighConf": adv_cw_highconf
            }

            for m in methods:
                pred = model(batch_adv[m]).argmax(1).item()
                if pred == single_label.item():
                    results[m]["correct"] += 1
                if pred == single_target.item():  # Track hit rate for runner_up
                    results[m]["targeted_success"] += 1

                # Capture one visual example that shows the difference
                if len(results[m]["examples"]) < 1:
                    results[m]["examples"].append({
                        "img": denormalize(batch_adv[m][0]),
                        "noise": denormalize(batch_adv[m][0] - single_img[0]),
                        "pred": pred, "true": single_label.item(), "target": single_target.item()
                    })

    # Reporting and Visualization
    print("\nHolistic Adversarial Investigation Summary:")
    for m in methods:
        acc = results[m]["correct"] / limit
        t_acc = results[m]["targeted_success"] / limit
        print(f"{m:15} | Acc: {acc:.4f} | Target Hit Rate: {t_acc:.4f}")

    fig, axes = plt.subplots(len(methods), 2, figsize=(8, 2.5 * len(methods)))
    for i, m in enumerate(methods):
        ex = results[m]["examples"][0] if results[m]["examples"] else results["Clean"]["examples"][0]
        axes[i, 0].imshow(ex["img"], cmap='gray')
        title_text = f"{m}\nPred: {ex['pred']} (True: {ex['true']})"
        if m in ["PGD-Targeted", "CW-HighConf"]:
            title_text += f"\nTargeted: {ex['target']}"
        axes[i, 0].set_title(title_text, fontsize=9)
        axes[i, 1].imshow(ex["noise"], cmap='bwr', vmin=-1, vmax=1)
        axes[i, 1].set_title(f"Noise Map", fontsize=9)
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.tight_layout()
    viz_path = args.output_dir / f"{OUTPUT_PREFIX}_holistic_adversarial_comparison.png"
    fig.savefig(viz_path, dpi=180)
    plt.close(fig)
    print(f"\nSaved holistic adversarial comparison figure: {viz_path}")

    # Update JSON
    report_path = args.output_dir / REPORT_FILENAME
    if report_path.exists():
        with open(report_path, 'r') as f:
            report_data = json.load(f)
    else:
        report_data = {}

    report_data["holistic_adversarial_evaluation"] = {
        m: {
            "accuracy": results[m]["correct"] / limit,
            "targeted_hit_rate": results[m]["targeted_success"] / limit
        } for m in methods
    }
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"Updated JSON report with holistic adversarial metrics: {report_path}")


def main_sampling(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / CHECKPOINT_FILENAME

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run training first (--run-train)."
        )

    # print("=" * 60)
    # print("Sampling-Based Random Perturbation Analysis")
    # print("=" * 60)

    model = load_checkpoint_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    test_dataset = torchvision.datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        transform=get_mnist_transform(),
        download=args.download_dataset,
    )



    def select_analysis_samples(
            model: nn.Module,
            dataset: torchvision.datasets.MNIST,
            num_samples: int = 10,
            device: torch.device = torch.device("cpu"),
            seed: int = 42,
            selection_strategy: str = 'correct'  # 'correct', 'random', 'stratified', 'hard'
    ) -> List[int]:
        np.random.seed(seed)
        torch.manual_seed(seed)

        all_indices = np.arange(len(dataset))
        selected = []

        if selection_strategy == 'random':
            # Random selection from entire test set
            if num_samples > len(dataset):
                print(f"Warning: Requested {num_samples} samples but dataset has only {len(dataset)}. Using all.")
                selected = all_indices.tolist()
            else:
                selected = np.random.choice(all_indices, num_samples, replace=False).tolist()

        elif selection_strategy == 'stratified':
            # Stratified sampling by true label
            label_to_indices = {}
            for idx, (_, label) in enumerate(dataset):
                label_to_indices.setdefault(label, []).append(idx)

            # Calculate samples per class
            n_classes = len(label_to_indices)
            samples_per_class = max(1, num_samples // n_classes)
            selected = []

            for label, indices in label_to_indices.items():
                if len(indices) <= samples_per_class:
                    selected.extend(indices)
                else:
                    selected.extend(np.random.choice(indices, samples_per_class, replace=False).tolist())

            # If we have fewer samples than requested, add random ones
            if len(selected) < num_samples:
                remaining = [idx for idx in all_indices if idx not in selected]
                if remaining:
                    needed = num_samples - len(selected)
                    additional = np.random.choice(remaining, min(needed, len(remaining)), replace=False)
                    selected.extend(additional.tolist())

        elif selection_strategy == 'hard':
            # Select hardest samples (lowest prediction confidence)
            model.eval()
            all_confidences = []

            with torch.no_grad():
                for idx in all_indices[:1000]:  # Limit to first 1000 for efficiency
                    image, _ = dataset[idx]
                    image_tensor = image.unsqueeze(0).to(device)

                    logits = model(image_tensor)
                    probs = torch.softmax(logits, dim=1)
                    confidence, _ = torch.max(probs, dim=1)

                    all_confidences.append((idx, confidence.item()))

            # Sort by confidence (ascending)
            all_confidences.sort(key=lambda x: x[1])
            selected = [idx for idx, _ in all_confidences[:num_samples]]

        else:  # Default: 'correct' strategy
            # Original behavior - only correctly predicted samples
            correct_indices = []

            # First, test batches to find correctly predicted ones
            batch_size = 256
            sampler = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

            model.eval()
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(sampler):
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    _, predictions = torch.max(logits, dim=1)
                    correct = predictions == labels

                    # Get indices of correct predictions
                    batch_correct = correct.cpu().numpy()
                    batch_indices = np.arange(batch_idx * batch_size,
                                              min((batch_idx + 1) * batch_size, len(dataset)))
                    correct_indices.extend(batch_indices[batch_correct])

                    if len(correct_indices) >= num_samples * 2:  # Select more for filtering
                        break

            # Randomly select from correctly predicted samples
            if len(correct_indices) < num_samples:
                print(f"Warning: Only {len(correct_indices)} correctly predicted samples, using all")
                selected = correct_indices
            else:
                selected = np.random.choice(correct_indices, num_samples, replace=False).tolist()

        # Get detailed information for selected samples
        print(f"Selected {len(selected)} samples using '{selection_strategy}' strategy:")
        print("-" * 60)

        correct_count = 0
        confidences = []

        for idx in selected:
            image, true_label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, prediction = torch.max(probs, dim=1)

            is_correct = prediction.item() == true_label
            if is_correct:
                correct_count += 1
            confidences.append(confidence.item())

            print(f"  Index {idx:4d}: True label={true_label}, "
                  f"Prediction={prediction.item()}, "
                  f"Confidence={confidence.item():.4f}, "
                  f"Correct={'Yes' if is_correct else 'No'}")

        # Print summary statistics
        print("-" * 60)
        print(f"Summary:")
        print(f"  Total samples: {len(selected)}")
        print(f"  Correct predictions: {correct_count} ({correct_count / len(selected) * 100:.1f}%)")
        print(
            f"  Incorrect predictions: {len(selected) - correct_count} ({(len(selected) - correct_count) / len(selected) * 100:.1f}%)")
        print(f"  Average confidence: {np.mean(confidences):.4f}")

        selected = [int(idx) for idx in selected]
        return selected

    def add_noise_to_image(
            image: torch.Tensor,
            noise_type: str = 'gaussian',
            noise_strength: float = 0.1
    ) -> torch.Tensor:
        noisy_image = image.clone()

        # if noise_strength == 0.0 :
        #     return noisy_image

        if noise_type == 'gaussian':
            # Gaussian noise
            noise = torch.randn_like(noisy_image) * noise_strength
            noisy_image = noisy_image + noise

        elif noise_type == 'uniform':
            # Uniform distribution noise
            noise = (torch.rand_like(noisy_image) * 2 - 1) * noise_strength
            noisy_image = noisy_image + noise

        elif noise_type == 'salt_and_pepper':
            # Salt and pepper noise
            mask = torch.rand_like(noisy_image) < noise_strength
            salt = torch.ones_like(noisy_image)  # Salt noise (maximum value)
            pepper = torch.zeros_like(noisy_image)  # Pepper noise (minimum value)

            # Randomly choose salt or pepper
            salt_mask = mask & (torch.rand_like(noisy_image) > 0.5)
            pepper_mask = mask & ~salt_mask

            noisy_image[salt_mask] = salt[salt_mask]
            noisy_image[pepper_mask] = pepper[pepper_mask]

        elif noise_type == 'speckle':
            noise = torch.randn_like(noisy_image) * noise_strength
            noisy_image = noisy_image + noisy_image * noise

        # Ensure pixel values are in valid range [0, 1]
        # noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

        return noisy_image

    def visualize_noisy_images(
            dataset: torchvision.datasets.MNIST,
            sample_index: int,
            noise_type: str,
            noise_strengths: np.ndarray,
            output_dir: Path,
            device: torch.device = torch.device("cpu"),
            num_l2_samples: int = 100
    ) -> None:
        original_image, true_label = dataset[sample_index]
        image_tensor = original_image.unsqueeze(0)

        # 2x5
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        if len(noise_strengths) != 10:
            if len(noise_strengths) < 10:
                noise_strengths = list(noise_strengths) + [noise_strengths[-1]] * (10 - len(noise_strengths))
            else:
                noise_strengths = noise_strengths[:10]

        for i, strength in enumerate(noise_strengths):
            noisy_image = add_noise_to_image(
                image=image_tensor,
                noise_type=noise_type,
                noise_strength=strength
            )

            l2_distances = []
            for _ in range(num_l2_samples):
                noisy_sample = add_noise_to_image(
                    image=image_tensor,
                    noise_type=noise_type,
                    noise_strength=strength
                )
                l2_distances.append(calculate_l2_perturbation(image_tensor, noisy_sample))

            avg_l2 = np.mean(l2_distances)
            std_l2 = np.std(l2_distances)

            noisy_np = noisy_image.squeeze(0).cpu().numpy()

            ax = axes[i]
            ax.imshow(noisy_np[0], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f'Original\nL2=0.000')
            else:
                ax.set_title(f'σ={strength:.3f}\nL2={avg_l2:.3f}±{std_l2:.3f}')
            ax.axis('off')

        plt.suptitle(f'Noise Type: {noise_type}\nSample Index: {sample_index}, True Label: {true_label}',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        output_path = output_dir / f"{OUTPUT_PREFIX}_noisy_images_{noise_type}_index_{sample_index}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  The noise image visualization has been saved: {output_path}")

    def calculate_l2_perturbation(original_image: torch.Tensor,
                                  noisy_image: torch.Tensor) -> float:
        # Flatten the images to vectors
        original_flat = original_image.flatten()
        noisy_flat = noisy_image.flatten()

        # Calculate L2 distance
        l2_distance = torch.norm(noisy_flat - original_flat, p=2).item()

        return l2_distance

    def monte_carlo_perturbation_analysis(
            model: nn.Module,
            dataset: torchvision.datasets.MNIST,
            sample_indices: List[int],
            noise_type: str = 'gaussian',
            noise_strength: float = 0.1,
            num_monte_carlo: int = 1000,
            device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:

        model.eval()

        all_sample_results = {}

        for idx in sample_indices:
            image, true_label = dataset[idx]
            image_tensor = image.unsqueeze(0)  # Add batch dimension [1, C, H, W]

            # Original prediction (no noise)
            with torch.no_grad():
                clean_logits = model(image_tensor.to(device))
                clean_probs = torch.softmax(clean_logits, dim=1)
                clean_confidence, clean_prediction = torch.max(clean_probs, dim=1)

            # Store Monte Carlo results
            monte_carlo_predictions = []
            monte_carlo_confidences = []
            monte_carlo_probs = []
            monte_carlo_l2_distances = []

            for _ in range(num_monte_carlo):
                # Add random noise
                noisy_image = add_noise_to_image(
                    image=image_tensor,
                    noise_type=noise_type,
                    noise_strength=noise_strength
                )

                l2_distance = calculate_l2_perturbation(image_tensor, noisy_image)
                monte_carlo_l2_distances.append(l2_distance)

                # Predict
                with torch.no_grad():
                    noisy_logits = model(noisy_image.to(device))
                    noisy_probs = torch.softmax(noisy_logits, dim=1)
                    confidence, prediction = torch.max(noisy_probs, dim=1)

                monte_carlo_predictions.append(prediction.item())
                monte_carlo_confidences.append(confidence.item())
                monte_carlo_probs.append(noisy_probs.cpu().numpy())

            # Convert to numpy arrays
            predictions_np = np.array(monte_carlo_predictions)
            confidences_np = np.array(monte_carlo_confidences)
            probs_np = np.vstack(monte_carlo_probs)  # Shape: [num_monte_carlo, num_classes]
            l2_distances_np = np.array(monte_carlo_l2_distances)

            # Calculate statistics
            accuracy = np.mean(predictions_np == true_label)
            mean_confidence = np.mean(confidences_np)
            std_confidence = np.std(confidences_np)
            mean_l2_distance = np.mean(l2_distances_np)
            std_l2_distance = np.std(l2_distances_np)

            # Calculate error prediction class distribution
            error_mask = predictions_np != true_label
            error_predictions = predictions_np[error_mask]
            error_distribution = {}

            if len(error_predictions) > 0:
                unique_errors, error_counts = np.unique(error_predictions, return_counts=True)
                for err_class, count in zip(unique_errors, error_counts):
                    error_distribution[int(err_class)] = {
                        "count": int(count),
                        "percentage": float(count / len(error_predictions))
                    }

            # Calculate mean and std of class probabilities
            mean_class_probs = np.mean(probs_np, axis=0)
            std_class_probs = np.std(probs_np, axis=0)

            # Store this sample's results
            all_sample_results[idx] = {
                "true_label": int(true_label),
                "clean_prediction": int(clean_prediction.item()),
                "clean_confidence": float(clean_confidence.item()),
                "accuracy": float(accuracy),
                "mean_confidence": float(mean_confidence),
                "std_confidence": float(std_confidence),
                "mean_l2_distance": float(mean_l2_distance),
                "std_l2_distance": float(std_l2_distance),
                "error_rate": 1.0 - float(accuracy),
                "error_distribution": error_distribution,
                "mean_class_probabilities": mean_class_probs.tolist(),
                "std_class_probabilities": std_class_probs.tolist(),
                "num_correct": int(np.sum(predictions_np == true_label)),
                "num_errors": int(np.sum(predictions_np != true_label))
            }

        # Calculate aggregate statistics for all samples
        all_accuracies = [result["accuracy"] for result in all_sample_results.values()]
        all_mean_confidences = [result["mean_confidence"] for result in all_sample_results.values()]
        all_error_rates = [result["error_rate"] for result in all_sample_results.values()]
        all_mean_l2_distances = [result["mean_l2_distance"] for result in all_sample_results.values()]
        all_std_l2_distances = [result["std_l2_distance"] for result in all_sample_results.values()]

        # Calculate overall error class distribution
        overall_error_distribution = Counter()
        for result in all_sample_results.values():
            for err_class, err_info in result["error_distribution"].items():
                overall_error_distribution[err_class] += err_info["count"]

        # Convert to percentages
        total_errors = sum(overall_error_distribution.values())
        if total_errors > 0:
            overall_error_percentages = {
                int(k): float(v / total_errors)
                for k, v in overall_error_distribution.items()
            }
        else:
            overall_error_percentages = {}

        # Find the most vulnerable samples (highest error rate)
        sorted_samples = sorted(
            all_sample_results.items(),
            key=lambda x: x[1]["error_rate"],
            reverse=True
        )

        most_vulnerable = []
        for idx, result in sorted_samples[:3]:  # Top 3 most vulnerable
            most_vulnerable.append({
                "index": int(idx),
                "true_label": result["true_label"],
                "error_rate": result["error_rate"],
                "accuracy": result["accuracy"]
            })

        # Return complete results
        overall_result = {
            "noise_type": noise_type,
            "noise_strength": float(noise_strength),
            "num_samples": len(sample_indices),
            "num_monte_carlo": num_monte_carlo,
            "overall_accuracy": float(np.mean(all_accuracies)),
            "overall_mean_confidence": float(np.mean(all_mean_confidences)),
            "overall_mean_l2_distance": float(np.mean(all_mean_l2_distances)),
            "overall_std_l2_distance": float(np.mean(all_std_l2_distances)),
            "overall_error_rate": float(np.mean(all_error_rates)),
            "accuracy_std": float(np.std(all_accuracies)),
            "confidence_std": float(np.std(all_mean_confidences)),
            "overall_error_distribution": overall_error_percentages,
            "most_vulnerable_samples": most_vulnerable,
            "per_sample_results": all_sample_results
        }

        # Print summary
        print(f"  Accuracy: {overall_result['overall_accuracy']:.4f}")
        print(f"  Error Rate: {overall_result['overall_error_rate']:.4f}")
        print(f"  Average Confidence: {overall_result['overall_mean_confidence']:.4f}")
        print(f"  Average L2 Distance: {overall_result['overall_mean_l2_distance']:.4f}")

        if most_vulnerable:
            print(f"  Most vulnerable sample: Index={most_vulnerable[0]['index']}, "
                  f"Error Rate={most_vulnerable[0]['error_rate']:.4f}")
        else:
            print("  Most vulnerable sample: Index=N/A, Error Rate=N/A")

        if overall_error_percentages:
            most_common_error = max(overall_error_percentages.items(), key=lambda x: x[1])
            print(f"  Most common error class: {most_common_error[0]} ({most_common_error[1]:.2%})")

        return overall_result

    def generate_noise_type_visualization(
            noise_type: str,
            results: Dict[str, Any],
            noise_strengths: np.ndarray,
            output_dir: Path
    ) -> None:

        # Prepare data
        accuracies = []
        confidences = []
        error_rates = []

        for strength in noise_strengths:
            key = f"{strength:.3f}"
            if key in results:
                accuracies.append(results[key]["overall_accuracy"])
                confidences.append(results[key]["overall_mean_confidence"])
                error_rates.append(results[key]["overall_error_rate"])

        # Create visualization charts
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Accuracy vs Noise Strength
        axes[0, 0].plot(noise_strengths, accuracies, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Noise Strength (std)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title(f'{noise_type.title()} Noise: Accuracy vs Noise Strength')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])

        # 2. Confidence vs Noise Strength
        axes[0, 1].plot(noise_strengths, confidences, 'g-o', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Noise Strength (std)')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].set_title(f'{noise_type.title()} Noise: Confidence vs Noise Strength')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])

        # 3. Error Rate vs Noise Strength
        axes[0, 2].plot(noise_strengths, error_rates, 'r-o', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Noise Strength (std)')
        axes[0, 2].set_ylabel('Error Rate')
        axes[0, 2].set_title(f'{noise_type.title()} Noise: Error Rate vs Noise Strength')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim([0, 1.05])

        # 4. Robustness Score (Area Under Curve)
        auc_accuracy = np.trapezoid(accuracies, noise_strengths)
        max_auc = noise_strengths[-1]  # Theoretical maximum AUC
        robustness_score = auc_accuracy / max_auc if max_auc > 0 else 0

        axes[1, 0].fill_between(noise_strengths, 0, accuracies, alpha=0.3, color='blue')
        axes[1, 0].plot(noise_strengths, accuracies, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Noise Strength (std)')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title(f'Robustness Score: {robustness_score:.3f}\n(AUC={auc_accuracy:.3f})')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Error Class Distribution (using the last noise strength)
        if noise_strengths.size > 0:
            last_strength = f"{noise_strengths[-1]:.3f}"
            if last_strength in results:
                error_dist = results[last_strength]["overall_error_distribution"]
                if error_dist:
                    error_classes = list(error_dist.keys())
                    error_percentages = [error_dist[c] for c in error_classes]

                    bars = axes[1, 1].bar(range(len(error_classes)), error_percentages,
                                          color='red', alpha=0.7)
                    axes[1, 1].set_xlabel('Error Class')
                    axes[1, 1].set_ylabel('Error Proportion')
                    axes[1, 1].set_title(f'Error Class Distribution (Noise Strength={noise_strengths[-1]:.2f})')
                    axes[1, 1].set_xticks(range(len(error_classes)))
                    axes[1, 1].set_xticklabels([str(c) for c in error_classes])
                    axes[1, 1].grid(True, alpha=0.3, axis='y')

                    # Add value labels
                    for bar, percentage in zip(bars, error_percentages):
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                                        f'{percentage:.1%}', ha='center', va='bottom')

        # 6. Confidence Distribution Box Plot
        confidence_data = []
        labels = []

        for i, strength in enumerate(noise_strengths[:4]):  # Only show first 4 noise strengths
            key = f"{strength:.3f}"
            if key in results:
                sample_confs = []
                for sample_result in results[key]["per_sample_results"].values():
                    sample_confs.append(sample_result["mean_confidence"])

                if sample_confs:
                    confidence_data.append(sample_confs)
                    labels.append(f'{strength:.2f}')

        if confidence_data:
            box = axes[1, 2].boxplot(confidence_data, tick_labels=labels, patch_artist=True)

            # Set box plot colors
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(box['boxes'], colors[:len(confidence_data)]):
                patch.set_facecolor(color)

            axes[1, 2].set_xlabel('Noise Strength')
            axes[1, 2].set_ylabel('Confidence')
            axes[1, 2].set_title('Confidence Distribution Box Plot')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save image
        output_path = output_dir / f"{OUTPUT_PREFIX}_noise_{noise_type}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Visualization chart saved: {output_path}")

    def generate_noise_comprehensive_report(
            all_results: Dict[str, Any],
            noise_types: List[str],
            noise_strengths: np.ndarray,
            sample_indices: List[int],
            args: argparse.Namespace
    ) -> Dict[str, Any]:
        report = {
            "analysis_summary": {
                "timestamp": np.datetime64('now').astype(str),
                "noise_types_analyzed": noise_types,
                "noise_strengths": noise_strengths.tolist(),
                "num_samples": len(sample_indices),
                "num_monte_carlo": args.num_monte_carlo,
                "sample_indices": [int(idx) for idx in sample_indices]
            },
            "noise_type_comparison": {},
            "robustness_scores": {},
            "vulnerability_analysis": {},
            "detailed_results": all_results
        }

        # Compare different noise types
        for noise_type in noise_types:
            if noise_type in all_results:
                type_results = all_results[noise_type]

                # Calculate robustness score
                accuracies = []
                for strength in noise_strengths:
                    key = f"{strength:.3f}"
                    if key in type_results:
                        accuracies.append(type_results[key]["overall_accuracy"])

                if accuracies:
                    auc = np.trapezoid(accuracies, noise_strengths[:len(accuracies)])
                    max_auc = noise_strengths[len(accuracies) - 1] if len(accuracies) > 0 else 1
                    robustness_score = auc / max_auc if max_auc > 0 else 0

                    report["robustness_scores"][noise_type] = {
                        "robustness_score": float(robustness_score),
                        "auc": float(auc),
                        "final_accuracy": float(accuracies[-1]) if accuracies else 0
                    }

                # Collect most vulnerable samples
                if noise_strengths.size > 0:
                    last_strength = f"{noise_strengths[-1]:.3f}"
                    if last_strength in type_results:
                        vulnerable = type_results[last_strength]["most_vulnerable_samples"]
                        report["vulnerability_analysis"][noise_type] = vulnerable

        # Find the most robust noise type
        if report["robustness_scores"]:
            most_robust = max(
                report["robustness_scores"].items(),
                key=lambda x: x[1]["robustness_score"]
            )
            report["analysis_summary"]["most_robust_noise_type"] = most_robust[0]
            report["analysis_summary"]["most_robust_score"] = most_robust[1]["robustness_score"]

        # Find the most vulnerable noise type
        if report["robustness_scores"]:
            least_robust = min(
                report["robustness_scores"].items(),
                key=lambda x: x[1]["robustness_score"]
            )
            report["analysis_summary"]["least_robust_noise_type"] = least_robust[0]
            report["analysis_summary"]["least_robust_score"] = least_robust[1]["robustness_score"]

        return report

    def print_noise_summary_report(report_data: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("Random Perturbation Analysis Summary Report")
        print("=" * 60)

        summary = report_data.get("analysis_summary", {})
        robustness_scores = report_data.get("robustness_scores", {})

        print(f"\nAnalysis Settings:")
        print(f"  • Noise Types: {', '.join(summary.get('noise_types_analyzed', []))}")
        print(f"  • Noise Strength Range: {summary.get('noise_strengths', [])[:3]}...")
        print(f"  • Sample Count: {summary.get('num_samples', 0)}")
        print(f"  • Monte Carlo Simulations: {summary.get('num_monte_carlo', 0)}")

        print(f"\nRobustness Scores:")
        for noise_type, scores in robustness_scores.items():
            print(f"  • {noise_type}: Score={scores.get('robustness_score', 0):.3f}, "
                  f"Final Accuracy={scores.get('final_accuracy', 0):.4f}")

        if "most_robust_noise_type" in summary:
            print(f"\nMost Robust Noise Type: {summary['most_robust_noise_type']} "
                  f"(Score: {summary['most_robust_score']:.3f})")

        if "least_robust_noise_type" in summary:
            print(f"Least Robust Noise Type: {summary['least_robust_noise_type']} "
                  f"(Score: {summary['least_robust_score']:.3f})")

        # Display vulnerability analysis
        vulnerability = report_data.get("vulnerability_analysis", {})
        if vulnerability:
            print(f"\nMost Vulnerable Samples (at maximum noise strength):")
            for noise_type, samples in vulnerability.items():
                if samples:
                    print(f"  {noise_type}: Index={samples[0].get('index', 'N/A')}, "
                          f"Error Rate={samples[0].get('error_rate', 0):.4f}")

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

    # Define perturbation types and intensities
    noise_types = ['gaussian', 'uniform', 'salt_and_pepper', 'speckle'] if args.noise_type == 'all' else [
        args.noise_type]
    noise_strengths = np.linspace(args.min_noise_std, args.max_noise_std, args.num_noise_levels)

    # Select analysis samples
    # print(f"\nSelecting analysis samples...")
    sample_indices = select_analysis_samples(
        model=model,
        dataset=test_dataset,
        num_samples=args.num_samples,
        device=device,
        seed=args.seed,
        selection_strategy=args.sample_strategy
    )

    all_results = {}

    for noise_type in noise_types:
        print(f"\n{'=' * 50}")
        print(f"Noise Type: {noise_type}")
        print(f"{'=' * 50}")

        type_results = {}

        for noise_strength in noise_strengths:
            print(f"\nNoise Strength: {noise_strength:.3f}")
            print("-" * 30)

            results = monte_carlo_perturbation_analysis(
                model=model,
                dataset=test_dataset,
                sample_indices=sample_indices,
                noise_type=noise_type,
                noise_strength=noise_strength,
                num_monte_carlo=args.num_monte_carlo,
                device=device
            )

            type_results[f"{noise_strength:.3f}"] = results

        all_results[noise_type] = type_results

        # Generate visualization for this noise type
        generate_noise_type_visualization(
            noise_type=noise_type,
            results=type_results,
            noise_strengths=noise_strengths,
            output_dir=args.output_dir
        )

        if len(sample_indices) > 0:
            sample_index = sample_indices[0]
            print(f"\nGenerate noise image visualization (index: {sample_index})...")

            visualize_noisy_images(
                dataset=test_dataset,
                sample_index=sample_index,
                noise_type=noise_type,
                noise_strengths=noise_strengths,
                output_dir=args.output_dir,
                device=device
            )

    # print(f"\n{'=' * 50}")
    # print("Generating Comprehensive Report")
    # print(f"{'=' * 50}")

    # Generate comprehensive report
    report_data = generate_noise_comprehensive_report(
        all_results=all_results,
        noise_types=noise_types,
        noise_strengths=noise_strengths,
        sample_indices=sample_indices,
        args=args
    )

    report_path = args.output_dir / f"{OUTPUT_PREFIX}_sampling_robustness_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\nAnalysis complete! Report saved to: {report_path}")
    # print_noise_summary_report(report_data)

if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.run_train:
        print("Running training pipeline...")
        main_train(cli_args)
    if cli_args.run_sensitivity:
        print("Running sensitivity analysis pipeline...")
        main_sensitivity(cli_args)
    if cli_args.run_robustness:
        print("Running robustness analysis pipeline...")
        main_robustness(cli_args)
    if cli_args.run_adversarial:
        print("Running adversarial attack pipeline...")
        main_adversarial(cli_args)
    if cli_args.run_sampling:
        print("Running sampling-based analysis pipeline...")
        main_sampling(cli_args)
    if not any([cli_args.run_train, cli_args.run_sensitivity, cli_args.run_robustness, cli_args.run_adversarial,
                cli_args.run_sampling]):
        print(
            "No action specified. Use --run-train, --run-sensitivity, --run-robustness, --run-adversarial, or --run-sampling.")
