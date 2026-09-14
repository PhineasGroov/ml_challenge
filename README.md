# Projet Machine Learning : Colorisation d'images (PyTorch)

Ce projet implémente une pipeline complète de colorisation automatique d'images en niveaux de gris vers la couleur en utilisant l'espace de couleur **CIELab** et des réseaux de neurones convolutifs sous **PyTorch**.

Le projet compare deux approches principales :
1. **Baseline U-Net + MSE** : Réseau U-Net personnalisé entraîné de zéro.
2. **Transfer Learning avec ResNet-18** : U-Net exploitant un encodeur ResNet-18 pré-entraîné sur ImageNet pour enrichir l'extraction sémantique et stabiliser l'apprentissage.

---

## 1. Installation et Dépendances

### Prérequis
- Python 3.10 ou supérieur
- Un GPU CUDA est recommandé pour l'entraînement (détecté automatiquement).

### Installation
Clonez le dépôt puis installez les dépendances :
```bash
git clone <URL_DU_REPO>
cd ml_challenge
pip install -r requirements.txt
```

---

## 2. Structure du Projet

```text
ml_challenge/
├── jpg/                     # Images d'entrainement et de test (Oxford 102 Flowers)
├── src/
│   ├── load.py              # Chargement, prétraitement Lab et découpage train/val/test
│   ├── model.py             # U-Net classique (DownBlock, UpBlock, UNet)
│   ├── model_resnet.py      # ResNetUNet (Backbone ResNet-18 + décodeur avec skip-connections)
│   ├── train.py             # Boucle d'entraînement, validation et sauvegarde
│   ├── evaluate.py          # Évaluation quantitative (MSE, Delta_ab, Dispersion)
│   ├── demo.py              # Génération de grilles comparatives d'images de test
│   ├── plot_history.py      # Tracé des courbes de loss train vs val
│   └── checkpoints/         # Sauvegarde des poids (.pth) et historiques (.json)
├── requirements.txt         # Dépendances Python
├── README.md                # Documentation d'utilisation
└── RAPPORT_GUIDE.md         # Guide méthodologique et théorique pour le rapport
```

---

## 3. Lancer un Entraînement

Tous les scripts s'exécutent depuis le dossier `src/`.

### A. Entraîner le U-Net Baseline
```bash
cd src
python train.py --model unet --epochs 20 --batch-size 32 --lr 1e-3
```
- Le meilleur modèle sera sauvegardé dans `src/checkpoints/unet_mse.pth`.
- L'historique des pertes sera stocké dans `src/checkpoints/history_unet.json`.

### B. Entraîner le U-Net avec Transfert d'Apprentissage (ResNet-18)
Par défaut, l'encodeur est **gelé** (recommandé pour un entraînement rapide et robuste sur les représentations ImageNet) :
```bash
cd src
python train.py --model resnet18 --epochs 20 --batch-size 32 --lr 1e-3
```

Pour débloquer tous les poids (fine-tuning complet) :
```bash
python train.py --model resnet18 --unfreeze-encoder --epochs 20 --lr 1e-4
```

---

## 4. Évaluation Quantitative (Test Set)

Conformément à la section 5 du sujet, le script `evaluate.py` calcule sur l'ensemble de test :
- La perte MSE normalisée
- L'erreur chromatique $\Delta ab = \sqrt{(a - \hat{a})^2 + (b - \hat{b})^2}$ en unités Lab réelles
- La dispersion (écart-type $\sigma_a, \sigma_b$) comparée à la vérité terrain

```bash
# Évaluer la baseline U-Net :
python evaluate.py --model unet --checkpoint checkpoints/unet_mse.pth

# Évaluer le modèle ResNet-18 :
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_mse.pth
```

---

## 5. Visualisation et Démonstration

### Générer une comparaison côte-à-côte (Baseline vs ResNet-18 vs Vérité Terrain)
```bash
python demo.py --compare --checkpoint-unet checkpoints/unet_mse.pth --checkpoint-resnet checkpoints/resnet18_mse.pth --out comparison_output.png
```

### Visualiser un modèle spécifique
```bash
python demo.py --model resnet18 --checkpoint checkpoints/resnet18_mse.pth --out demo_resnet.png
```

### Tracer les courbes d'apprentissage (Loss curves)
```bash
python plot_history.py checkpoints/history_unet.json checkpoints/history_resnet18.json --out loss_curves.png
```

---

## 6. Dataset utilisé
- Jeu de données : **Oxford 102 Flowers Dataset** (~8 189 images).
- Séparation : 80% Entraînement, 10% Validation, 10% Test (avec graine aléatoire fixe pour reproductibilité).
