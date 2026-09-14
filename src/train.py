"""
Boucle d'entrainement pour la premiere approche : U-Net + MSE.

On assemble ici les briques deja construites :
- load.py  : dataset et dataloaders
- model.py : architecture U-Net
"""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from load import ColorizationDataset, list_image_files, split_files
from model import UNet


def get_device() -> torch.device:
    # TODO J : renvoyer torch.device("cuda") si torch.cuda.is_available(),
    # sinon torch.device("cpu"). (Indice : torch.cuda.is_available() renvoie un bool)
    if torch.cuda.is_available() :
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(
    root: str,
    image_size: int = 128,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construit les DataLoader train / val / test."""
    # TODO K :
    #   1. files = list_image_files(root)
    #   2. train_files, val_files, test_files = split_files(files)
    #   3. creer les 3 ColorizationDataset (un par liste de fichiers)
    #   4. creer les 3 DataLoader correspondants. Attention :
    #      - shuffle=True uniquement pour le train (val/test : shuffle=False,
    #        pas besoin de melanger puisqu'on ne met pas a jour les poids)
    #      - meme batch_size pour les 3 (par simplicite)
    #   5. renvoyer (train_loader, val_loader, test_loader)
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

    parser = argparse.ArgumentParser(description="Entrainement U-Net pour la colorisation")
    parser.add_argument("--root", type=str, default="..", help="Dossier racine contenant jpg/")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unet_mse.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Device : {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        args.root, image_size=args.image_size, batch_size=args.batch_size
    )
    print(f"Train: {len(train_loader.dataset)} images | "
          f"Val: {len(val_loader.dataset)} images | "
          f"Test: {len(test_loader.dataset)} images")

    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d}/{args.epochs} - "
              f"train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> nouveau meilleur modele sauvegarde ({checkpoint_path})")

    print(f"Entrainement termine. Meilleure val_loss : {best_val_loss:.5f}")


if __name__ == "__main__":
    main()
