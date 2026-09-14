"""
Evaluation quantitative d'un modele de colorisation sur l'ensemble de test.

Metriques calculees (Section 5 du sujet) :
1. MSE (sur les tenseurs normalises)
2. Delta_ab moyen : distance euclidienne chromatique dans l'espace Lab reel (CIE76)
   Delta_ab = sqrt((a - a_pred)^2 + (b - b_pred)^2)
3. Dispersion (ecart-type std_a, std_b) des valeurs reelles vs predites,
   permettant de mesurer si le modele s'effondre vers des teintes grises / ternes.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from load import ColorizationDataset, list_image_files, split_files
from model import UNet
try:
    from model_resnet import ResNetUNet
except ImportError:
    ResNetUNet = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint_path: str,
    model_type: str = "unet",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Charge le modele selon le type specifie ('unet' ou 'resnet18')."""
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


def evaluate_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Calcule l'ensemble des metriques sur le test_loader."""
    model.eval()
    mse_criterion = nn.MSELoss()

    total_mse = 0.0
    total_delta_ab = 0.0
    n_samples = 0

    all_a_real = []
    all_b_real = []
    all_a_pred = []
    all_b_pred = []

    with torch.no_grad():
        for L, ab in test_loader:
            L = L.to(device)
            ab = ab.to(device)
            batch_size = L.size(0)

            ab_pred = model(L)

            # 1. MSE sur tenseurs [-1, 1]
            loss_mse = mse_criterion(ab_pred, ab)
            total_mse += loss_mse.item() * batch_size

            # Dénormalisation vers l'espace Lab réel (ab in [-128, 127])
            # ab a la forme (B, 2, H, W) -> canal 0 = a, canal 1 = b
            a_real = ab[:, 0, :, :].cpu().numpy() * 128.0
            b_real = ab[:, 1, :, :].cpu().numpy() * 128.0
            a_pred = ab_pred[:, 0, :, :].cpu().numpy() * 128.0
            b_pred = ab_pred[:, 1, :, :].cpu().numpy() * 128.0

            # 2. Delta ab moyen (par pixel puis moyenne par image)
            delta_ab = np.sqrt((a_real - a_pred) ** 2 + (b_real - b_pred) ** 2)
            # moyenne sur H, W pour chaque image du batch
            total_delta_ab += np.mean(delta_ab) * batch_size

            # 3. Stocker pour calculer l'ecart-type (dispersion) global
            all_a_real.append(a_real.reshape(-1))
            all_b_real.append(b_real.reshape(-1))
            all_a_pred.append(a_pred.reshape(-1))
            all_b_pred.append(b_pred.reshape(-1))

            n_samples += batch_size

    # Assemblage de la dispersion
    concat_a_real = np.concatenate(all_a_real)
    concat_b_real = np.concatenate(all_b_real)
    concat_a_pred = np.concatenate(all_a_pred)
    concat_b_pred = np.concatenate(all_b_pred)

    metrics = {
        "mse": total_mse / n_samples,
        "delta_ab": float(total_delta_ab / n_samples),
        "std_a_real": float(np.std(concat_a_real)),
        "std_b_real": float(np.std(concat_b_real)),
        "std_a_pred": float(np.std(concat_a_pred)),
        "std_b_pred": float(np.std(concat_b_pred)),
        "mean_a_pred": float(np.mean(concat_a_pred)),
        "mean_b_pred": float(np.mean(concat_b_pred)),
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation quantitative de la colorisation")
    parser.add_argument("--root", type=str, default="..", help="Dossier racine contenant jpg/")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unet_mse.pth", help="Fichier .pth")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resnet18"], help="Type d'architecture")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    print(f"Evaluation sur l'appareil : {device}")

    if not Path(args.checkpoint).exists():
        print(f"Erreur : Le checkpoint '{args.checkpoint}' n'existe pas.")
        return

    print(f"Chargement du modele ({args.model}) depuis {args.checkpoint}...")
    model = load_model(args.checkpoint, model_type=args.model, device=device)

    files = list_image_files(args.root)
    _, _, test_files = split_files(files)
    print(f"Ensemble de test : {len(test_files)} images.")

    test_dataset = ColorizationDataset(test_files, image_size=args.image_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_metrics(model, test_loader, device)

    print("\n" + "=" * 55)
    print("           RESULTATS DE L'EVALUATION (TEST SET)")
    print("=" * 55)
    print(f"  MSE (normalisee [-1, 1])  : {metrics['mse']:.5f}")
    print(f"  Delta_ab moyen (Lab reel) : {metrics['delta_ab']:.2f}")
    print("-" * 55)
    print("  Dispersion / Saturation (Ecart-type standard) :")
    print(f"    - Canal 'a' reel   : {metrics['std_a_real']:.2f}  | predit : {metrics['std_a_pred']:.2f}")
    print(f"    - Canal 'b' reel   : {metrics['std_b_real']:.2f}  | predit : {metrics['std_b_pred']:.2f}")
    print("-" * 55)
    print("  Centrage moyen des predictions :")
    print(f"    - Moyenne 'a' predite : {metrics['mean_a_pred']:+.2f}")
    print(f"    - Moyenne 'b' predite : {metrics['mean_b_pred']:+.2f}")
    print("=" * 55)
    print("\nInterpretation pour le rapport :")
    print(" Si std(a_pred) et std(b_pred) sont tres inferieurs aux reels,")
    print(" cela quantifie rigoureusement l'affadissement des couleurs du aux proprietes de la MSE.\n")


if __name__ == "__main__":
    main()
