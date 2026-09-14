"""
Demo / visualisation : compare l'image reelle a l'image colorisee par le(s) modele(s),
sur des images de test jamais vues pendant l'entrainement.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from skimage.color import lab2rgb

from load import ColorizationDataset, list_image_files, split_files
from model import UNet
try:
    from model_resnet import ResNetUNet
except ImportError:
    ResNetUNet = None
from train import get_device


def denormalize_lab(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Inverse la normalisation appliquee dans ColorizationDataset.

    L   : tableau (H, W), valeurs dans [-1, 1]
    ab  : tableau (H, W, 2), valeurs dans [-1, 1]
    Renvoie un tableau Lab (H, W, 3) avec les vraies plages (L in [0,100], ab in [-128,127]).
    """
    L_real = (L + 1) * 50
    ab_real = ab * 128
    L_real = np.expand_dims(L_real, axis=-1)
    return np.concatenate([L_real, ab_real], axis=-1)


def load_model(
    checkpoint_path: str,
    model_type: str = "unet",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Charge un modele (UNet ou ResNetUNet) entraine a partir d'un fichier .pth."""
    if model_type == "unet":
        model = UNet()
    elif model_type == "resnet18":
        if ResNetUNet is None:
            raise ImportError("model_resnet.py introuvable.")
        model = ResNetUNet()
    else:
        raise ValueError(f"Type de modele inconnu : {model_type}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def colorize(model: nn.Module, L_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Fait l'inference sur un seul exemple L (1,H,W) et renvoie l'image RGB predite."""
    with torch.no_grad():
        L_batch = L_tensor.unsqueeze(0).to(device)  # (1,1,H,W)
        ab_pred = model(L_batch)                    # (1,2,H,W)

    L_np = L_tensor.squeeze(0).cpu().numpy()          # (H,W)
    ab_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,2)

    lab = denormalize_lab(L_np, ab_np)
    rgb = lab2rgb(lab)
    return np.clip(rgb, 0, 1)


def show_comparisons(
    root: str = "..",
    checkpoint_path: str = "checkpoints/unet_mse.pth",
    model_type: str = "unet",
    checkpoint_resnet: str = "checkpoints/resnet18_mse.pth",
    compare: bool = False,
    image_size: int = 128,
    n_examples: int = 6,
    seed: int = 0,
    out_path: str = "demo_output.png",
) -> None:
    device = get_device()

    files = list_image_files(root)
    _, _, test_files = split_files(files)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(test_files, size=n_examples, replace=False)
    dataset = ColorizationDataset(list(chosen), image_size=image_size)

    if compare:
        # Mode comparaison : Grayscale | Verite terrain | U-Net Baseline | ResNet-18 Transfert
        print(f"Chargement du U-Net baseline ({checkpoint_path})...")
        model_unet = load_model(checkpoint_path, model_type="unet", device=device)
        print(f"Chargement du ResNet-18 ({checkpoint_resnet})...")
        model_resnet = load_model(checkpoint_resnet, model_type="resnet18", device=device)

        n_cols = 4
        col_titles = ["Entree (Gris)", "Verite terrain", "Baseline (U-Net)", "Transfert (ResNet-18)"]
        fig, axes = plt.subplots(n_examples, n_cols, figsize=(12, 3 * n_examples))

        for i in range(n_examples):
            L_tensor, ab_tensor = dataset[i]
            gray = (L_tensor.squeeze(0).numpy() + 1) / 2
            real_lab = denormalize_lab(L_tensor.squeeze(0).numpy(), ab_tensor.permute(1, 2, 0).numpy())
            real_rgb = np.clip(lab2rgb(real_lab), 0, 1)

            pred_unet = colorize(model_unet, L_tensor, device)
            pred_resnet = colorize(model_resnet, L_tensor, device)

            axes[i, 0].imshow(gray, cmap="gray")
            axes[i, 1].imshow(real_rgb)
            axes[i, 2].imshow(pred_unet)
            axes[i, 3].imshow(pred_resnet)
            for j in range(n_cols):
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(col_titles[j], fontsize=12, fontweight="bold")
    else:
        # Mode simple 3 colonnes
        print(f"Chargement du modele {model_type} ({checkpoint_path})...")
        model = load_model(checkpoint_path, model_type=model_type, device=device)

        n_cols = 3
        col_titles = ["Entree (niveaux de gris)", "Verite terrain", f"Pred. ({model_type})"]
        fig, axes = plt.subplots(n_examples, n_cols, figsize=(9, 3 * n_examples))

        for i in range(n_examples):
            L_tensor, ab_tensor = dataset[i]
            gray = (L_tensor.squeeze(0).numpy() + 1) / 2
            real_lab = denormalize_lab(L_tensor.squeeze(0).numpy(), ab_tensor.permute(1, 2, 0).numpy())
            real_rgb = np.clip(lab2rgb(real_lab), 0, 1)

            pred_rgb = colorize(model, L_tensor, device)

            axes[i, 0].imshow(gray, cmap="gray")
            axes[i, 1].imshow(real_rgb)
            axes[i, 2].imshow(pred_rgb)
            for j in range(n_cols):
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(col_titles[j], fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Figure sauvegardee avec succes dans '{out_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstration visuelle de colorisation")
    parser.add_argument("--root", type=str, default="..", help="Dossier racine contenant jpg/")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resnet18"], help="Type de modele a tester")
    parser.add_argument("--checkpoint", type=str, default=None, help="Chemin du checkpoint")
    parser.add_argument("--compare", action="store_true", help="Active la comparaison cote-a-cote U-Net vs ResNet-18")
    parser.add_argument("--checkpoint-unet", type=str, default="checkpoints/unet_mse.pth")
    parser.add_argument("--checkpoint-resnet", type=str, default="checkpoints/resnet18_mse.pth")
    parser.add_argument("--n-examples", type=int, default=6, help="Nombre d'images a afficher")
    parser.add_argument("--seed", type=int, default=0, help="Graine pour selectionner les images de test fixes")
    parser.add_argument("--out", type=str, default="demo_output.png", help="Chemin de sortie PNG")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = f"checkpoints/{args.model}_mse.pth"

    show_comparisons(
        root=args.root,
        checkpoint_path=checkpoint if not args.compare else args.checkpoint_unet,
        model_type=args.model,
        checkpoint_resnet=args.checkpoint_resnet,
        compare=args.compare,
        n_examples=args.n_examples,
        seed=args.seed,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

