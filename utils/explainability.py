"""SHAP and Grad-CAM explainability for MMRCCNet."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch


def generate_shap_clinical(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    device: torch.device,
) -> Path:
    """SHAP values for clinical branch (Figure 3)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    def predict_fn(x):
        t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(clinical=t)
            return torch.sigmoid(out["metastasis_logit"]).cpu().numpy()

    background = shap.sample(X_background, min(200, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain[:100], nsamples=100)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain[:100], feature_names=feature_names, show=False)
    out_path = output_dir / "shap_clinical_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    plt.barh([feature_names[i] for i in order], mean_abs[order], color="steelblue")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Clinical Feature Importance (SHAP)")
    plt.gca().invert_yaxis()
    bar_path = output_dir / "shap_clinical_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def generate_gradcam_imaging(
    model,
    image_tensor: torch.Tensor,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Grad-CAM for imaging branch (Figure 4)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    image_tensor = image_tensor.to(device)
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)  # (1, 1, H, W)

    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_layer = model.get_imaging_gradcam_target()
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    out = model(imaging=image_tensor.squeeze(1))
    score = out["metastasis_logit"]
    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    act = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    img = image_tensor[0, 0].cpu().numpy()
    from scipy.ndimage import zoom
    cam_up = zoom(cam, (img.shape[0] / cam.shape[0], img.shape[1] / cam.shape[1]), order=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("CT Slice")
    axes[0].axis("off")
    axes[1].imshow(cam_up, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(cam_up, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    out_path = output_dir / "gradcam_imaging.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
