"""
Architecture U-Net pour la colorisation.

On construit le reseau bloc par bloc :
- DownBlock : 2x (Conv2D + BatchNorm + ReLU) puis MaxPool (Figure 3 du sujet)
- UpBlock   : ConvTranspose2D, concatenation avec le skip, puis 2x (Conv2D + BatchNorm + ReLU) (Figure 4 du sujet)
- UNet      : assemblage complet avec les skip-connections (Figure 5/6 du sujet)
"""

import torch
from torch import nn


class DownBlock(nn.Module):
    """Bloc descendant : extrait des features et reduit la resolution par 2.

    Renvoie deux tenseurs :
      - `skip` : le resultat juste avant le pooling (a transmettre au bloc montant miroir)
      - `pooled` : le resultat apres MaxPooling (a transmettre au bloc descendant suivant)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # TODO A : declarer les couches necessaires (voir Figure 3) :
        #   - self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #   - self.bn1   = nn.BatchNorm2d(out_channels)
        #   - self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #   - self.bn2   = nn.BatchNorm2d(out_channels)
        #   - self.relu  = nn.ReLU(inplace=True)
        #   - self.pool  = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO B : enchainer conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu
        # stocker ce resultat dans `skip`, puis appliquer le pooling dessus
        # pour obtenir `pooled`. Renvoyer (skip, pooled).
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2)
        skip = self.relu(bn2)

        pooled = self.pool(skip)

        return skip, pooled


class UpBlock(nn.Module):
    """Bloc montant : augmente la resolution par 2 et fusionne avec le skip.

    Parametres :
      - in_channels   : nombre de canaux du tenseur entrant (venant du bloc precedent)
      - skip_channels : nombre de canaux du tenseur `skip` (venant du DownBlock miroir)
      - out_channels  : nombre de canaux en sortie de ce bloc
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # TODO C : declarer les couches necessaires (voir Figure 4) :
        #   - self.up    = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        #     (la transposee ramene direct au nombre de canaux `out_channels`)
        #   - self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        #     (l'entree de conv1 = canaux post-transposee + canaux du skip concatene)
        #   - self.bn1   = nn.BatchNorm2d(out_channels)
        #   - self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #   - self.bn2   = nn.BatchNorm2d(out_channels)
        #   - self.relu  = nn.ReLU(inplace=True)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # TODO D :
        #   1. x_up = self.up(x)                                  -> augmente H,W
        #   2. merged = torch.cat([x_up, skip], dim=1)            -> concatene sur les canaux
        #   3. enchainer conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu sur `merged`
        #   4. renvoyer le resultat final
        x_up = self.up(x)
        merged = torch.cat([x_up, skip], dim = 1)
        conv1  = self.conv1(merged)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2)
        out = self.relu(bn2)

        return out


class UNet(nn.Module):
    """U-Net complet pour la colorisation : L (1,H,W) -> ab (2,H,W).

    Architecture a 3 niveaux + goulet :
      1 -> 32 -> 64 -> 128 -> (goulet 256) -> 128 -> 64 -> 32 -> 2
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()

        # TODO E : declarer les 3 DownBlock (voir les tailles de canaux ci-dessus)
        self.down1 = DownBlock(in_channels, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)

        # TODO F : declarer le goulet : 2 convs (128->256, puis 256->256),
        # chacune suivie d'un BatchNorm2d et d'un ReLU. Pas de pooling ici,
        # c'est le point le plus profond du reseau, il ne reduit plus la resolution.
        self.bottleneck_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottleneck_bn1   = nn.BatchNorm2d(256)
        self.bottleneck_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bottleneck_bn2   = nn.BatchNorm2d(256)
        self.relu             = nn.ReLU(inplace=True)

        # TODO G : declarer les 3 UpBlock (attention a l'ordre des arguments :
        # in_channels, skip_channels, out_channels). Le premier UpBlock recoit
        # la sortie du goulet (256 canaux) et le skip du DownBlock3 (128 canaux) :
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)

        # TODO H : declarer la couche de sortie finale :
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.tanh     = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO I : propager x a travers le reseau complet.
        skip1, pooled1 = self.down1(x)
        skip2, pooled2 = self.down2(pooled1)
        skip3, pooled3 = self.down3(pooled2)
        #   4. goulet : appliquer conv1->bn1->relu->conv2->bn2->relu sur pooled3
        goulet1 = self.bottleneck_conv1(pooled3)
        goulet1_bn1 = self.bottleneck_bn1(goulet1)
        goulet1 = self.relu(goulet1_bn1)

        goulet2 = self.bottleneck_conv2(goulet1)
        goulet2_bn2 = self.bottleneck_bn2(goulet2)
        goulet2 = self.relu(goulet2_bn2)

        #   5. remontee : up3(goulet, skip3) -> up2(...) -> up1(...)
        up3 = self.up3(goulet2, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)

        #   6. sortie : out_conv puis tanh
        out_conv = self.out_conv(up1)
        output = self.tanh(out_conv)
        #   7. renvoyer le resultat, de forme (B, 2, H, W)

        return output