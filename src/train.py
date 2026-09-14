"""
Boucle d'entrainement pour la premiere approche : U-Net + MSE.

On assemble ici les briques deja construites :
- load.py  : dataset et dataloaders
- model.py : architecture U-Net
"""

from pathlib import Path
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from load import ColorizationDataset, list_image_files, split_files
from model import UNet
from model_resnet import ResNetUNet


def get_device() -> torch.device:
    # Renvoie torch.device("cuda") si torch.cuda.is_available(), sinon cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(
    root: str,
    image_size: int = 128,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construit les DataLoader train / val / test."""
    files = list_image_files(root)
    train_files, val_files, test_files = split_files(files)
    train_dataset = ColorizationDataset(train_files, image_size=image_size)
    val_dataset = ColorizationDataset(val_files, image_size=image_size)
    test_dataset = ColorizationDataset(test_files, image_size=image_size)
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data, val_data, test_data


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Effectue une epoque d'entrainement complete, renvoie la loss moyenne."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for L, ab in loader:
        L = L.to(device)
        ab = ab.to(device)

        optimizer.zero_grad()
        ab_pred = model(L)
        loss = criterion(ab_pred, ab)
        loss.backward()
        optimizer.step()

        batch_size = L.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evalue le modele sur `loader` (val ou test), sans mise a jour des poids."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for L, ab in loader:
            L = L.to(device)
            ab = ab.to(device)

            ab_pred = model(L)
            loss = criterion(ab_pred, ab)

            batch_size = L.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Entrainement du modele pour la colorisation")
    parser.add_argument("--root", type=str, default="..", help="Dossier racine contenant jpg/")
    parser.add_argument("--model", type=str, default="resnet18", choices=["unet", "resnet18"],
                        help="Architecture a entrainer : 'unet' (baseline) ou 'resnet18' (transfer learning)")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Degele l'encodeur ResNet pour entrainer tous les poids (par defaut l'encodeur est gele)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin de sauvegarde du modele (par defaut checkpoints/<model>_mse.pth)")
    parser.add_argument("--history", type=str, default=None,
                        help="Chemin de sauvegarde de l'historique JSON des pertes")
    args = parser.parse_args()

    device = get_device()
    print(f"Appareil utilise (Device) : {device}")

    # Choix du nom par defaut des checkpoints
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model}_mse.pth"
    if args.history is None:
        args.history = f"checkpoints/history_{args.model}.json"

    train_loader, val_loader, test_loader = build_dataloaders(
        args.root, image_size=args.image_size, batch_size=args.batch_size
    )
    print(f"Train: {len(train_loader.dataset)} images | "
          f"Val: {len(val_loader.dataset)} images | "
          f"Test: {len(test_loader.dataset)} images")

    # Instanciation de l'architecture choisie
    if args.model == "unet":
        print("Architecture : U-Net classique (entraine a partir de zero)")
        model = UNet().to(device)
    elif args.model == "resnet18":
        freeze = not args.unfreeze_encoder
        status_str = "gele (freeze)" if freeze else "non gele (fine-tuning complet)"
        print(f"Architecture : ResNet-18 U-Net (Pre-entraine sur ImageNet, encodeur {status_str})")
        model = ResNetUNet(pretrained=True, freeze_encoder=freeze).to(device)

    criterion = nn.MSELoss()
    # On n'optimise que les parametres qui requierent un gradient (particulierement utile si l'encodeur est gele)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    history = {
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_loss": [],
        "val_loss": [],
    }

    print(f"\nDebut de l'entrainement pour {args.epochs} epoques...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} - "
              f"train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> nouveau meilleur modele sauvegarde ({checkpoint_path})")

    # Sauvegarde de l'historique JSON pour tracer les courbes de loss dans le rapport
    with open(args.history, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Historique d'entrainement enregistre dans {args.history}")
    print(f"Entrainement termine. Meilleure val_loss : {best_val_loss:.5f}\n")


if __name__ == "__main__":
    main()

