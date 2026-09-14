"""
Script utilitaire pour tracer les courbes de perte (Loss curves) train vs val
a partir des fichiers d'historique JSON sauvegardes par train.py.
"""

from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt


def plot_histories(history_files: list[str], out_path: str = "loss_curves.png") -> None:
    plt.figure(figsize=(10, 6))

    for h_path in history_files:
        p = Path(h_path)
        if not p.exists():
            print(f"Fichier {h_path} introuvable, ignore.")
            continue

        with open(p, "r") as f:
            data = json.load(f)

        model_name = data.get("model", p.stem)
        train_loss = data.get("train_loss", [])
        val_loss = data.get("val_loss", [])
        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, label=f"{model_name} (Train Loss)", linestyle="--")
        plt.plot(epochs, val_loss, label=f"{model_name} (Val Loss)", linewidth=2)

    plt.xlabel("Epoque", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Evolution des fonctions de perte (Train vs Validation)", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Graphique des courbes de loss sauvegarde dans '{out_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tracer les courbes d'entrainement")
    parser.add_argument("histories", nargs="*", default=[
        "checkpoints/history_unet.json",
        "checkpoints/history_resnet18.json"
    ], help="Fichiers JSON d'historique a tracer")
    parser.add_argument("--out", type=str, default="loss_curves.png", help="Chemin de sortie PNG")
    args = parser.parse_args()

    plot_histories(args.histories, out_path=args.out)


if __name__ == "__main__":
    main()
