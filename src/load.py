from pathlib import Path
from PIL import Image
from skimage.color import rgb2lab

import numpy as np
import random
import torch
from torch.utils.data import Dataset


def list_image_files(root: str) -> list[Path]:
    data_dir = Path(f'{root}/jpg')
    img_list = list(data_dir.glob('*.jpg'))

    return sorted(img_list)


def split_files(
    files: list[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Separe la liste `files` en trois listes : train, val, test."""
    # TODO 2 :
    #   - copier la liste et la melanger de facon reproductible (voir le
    #     module `random`, avec random.Random(seed).shuffle(...))
    #   - calculer les indices de coupure a partir de train_ratio et val_ratio
    #   - trancher la liste en 3 sous-listes et les renvoyer
    files_copy = list(files) #pour conserver l'original des fichiers
    random.Random(seed).shuffle(files_copy)
    n_train = int(len(files_copy)*train_ratio)
    n_val = n_train + int(len(files_copy)*val_ratio)

    train = files_copy[:n_train]
    val = files_copy[n_train:n_val]
    test = files_copy[n_val:]

    return train, val, test


class ColorizationDataset(Dataset):
    """Dataset renvoyant (L, ab) normalises a partir d'images RGB."""

    def __init__(self, files: list[Path], image_size: int = 128):
        self.files = files
        self.image_size = image_size

    def __len__(self) -> int:
        # TODO 3 : combien d'exemples contient ce dataset ?
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO 4 (le plus important, on le fera en plusieurs sous-etapes) :
        #   a. charger l'image self.files[idx] en RGB avec PIL
        #   b. la redimensionner a (self.image_size, self.image_size)
        #   c. la convertir en tableau numpy flottant dans [0, 1]
        #   d. la convertir en Lab avec skimage.color.rgb2lab
        #   e. separer L (canal 0) et ab (canaux 1 et 2)
        #   f. normaliser L vers [-1, 1] (L est dans [0, 100])
        #   g. normaliser ab vers [-1, 1] (a, b sont environ dans [-128, 127])
        #   h. construire les tenseurs torch de shape (1, H, W) pour L
        #      et (2, H, W) pour ab, en float32
        img = Image.open(self.files[idx]).convert('RGB').resize((self.image_size,self.image_size))
        arr = np.array(img).astype(np.float32) / 255.0

        lab = rgb2lab(arr)
        L_lab = lab[:, :, 0]
        a_lab = lab[:, :, 1]
        b_lab = lab[:, :, 2]

        L_conv = L_lab/50 - 1
        a_conv, b_conv = a_lab/128, b_lab/128

        L_tensor = np.expand_dims(L_conv, axis=0)
        ab_tensor = np.stack([a_conv, b_conv], axis=0)

        L = torch.from_numpy(L_tensor).float()
        ab = torch.from_numpy(ab_tensor).float()

        return L, ab



        