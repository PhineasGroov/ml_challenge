"""
Architecture U-Net avec encodeur pre-entraine ResNet-18 (Transfer Learning).
Section 6.1 du sujet :
- Encodeur : ResNet-18 pre-entraine sur ImageNet.
- Possibilite de geler les poids de l'encodeur (freeze).
- Decodeur : Blocs montants avec convolutions transposees et skip-connections.
- Sortie : Tanh pour borner les predictions dans [-1, 1].
"""

import torch
from torch import nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """Bloc de decodage (montant) :
    1. Augmentation de resolution spatiale (x2) via ConvTranspose2d
    2. Concatenation avec la skip-connection venant de l'encodeur
    3. Deux convolutions (3x3) avec BatchNorm et ReLU pour affiner les representations
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        # Gestion si H, W different legerement a cause d'arrondis
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = nn.functional.interpolate(x_up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([x_up, skip], dim=1)
        out = self.relu(self.bn1(self.conv1(merged)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class ResNetUNet(nn.Module):
    """U-Net utilisant un backbone ResNet-18 comme encodeur.

    Entree : (B, 1, H, W) - canal L normalise dans [-1, 1]
    Sortie : (B, 2, H, W) - canaux a, b normalises dans [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        pretrained: bool = True,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # 1. Chargement de ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        # 2. Adaptation de la premiere convolution pour accepter 1 canal (L)
        # Au lieu d'ecraser les filtres ImageNet, on moyenne les poids des 3 canaux RGB
        # afin de conserver la reponse optimale aux contours et textures en niveaux de gris.
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if pretrained:
                # Moyenne des poids sur la dimension des 3 canaux RGB
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            resnet.conv1 = new_conv

        # Deconstruction de l'encodeur ResNet
        self.encoder_init = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )  # Sortie : (64, H/2, W/2) -> skip 1

        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # Sortie : (64, H/4, W/4)  -> skip 2
        self.layer2 = resnet.layer2  # Sortie : (128, H/8, W/8) -> skip 3
        self.layer3 = resnet.layer3  # Sortie : (256, H/16, W/16) -> skip 4
        self.layer4 = resnet.layer4  # Sortie : (512, H/32, W/32) -> bottleneck

        # Gel eventuel de l'encodeur
        if freeze_encoder:
            self.freeze_encoder()

        # 3. Decodeur
        # Bottleneck (512, H/32, W/32) + Skip 4 (256, H/16, W/16) -> (256, H/16, W/16)
        self.up4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        # (256, H/16, W/16) + Skip 3 (128, H/8, W/8) -> (128, H/8, W/8)
        self.up3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        # (128, H/8, W/8) + Skip 2 (64, H/4, W/4) -> (64, H/4, W/4)
        self.up2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # (64, H/4, W/4) + Skip 1 (64, H/2, W/2) -> (64, H/2, W/2)
        self.up1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Derniere etape de remontee vers la resolution pleine (H, W)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),           # (2, H, W)
            nn.Tanh(),  # Sortie dans [-1, 1]
        )

    def freeze_encoder(self) -> None:
        """Gele les parametres de l'encodeur pour ne pas les modifier pendant l'entrainement."""
        encoder_parts = [self.encoder_init, self.layer1, self.layer2, self.layer3, self.layer4]
        for part in encoder_parts:
            for param in part.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Degele les parametres de l'encodeur pour le fine-tuning."""
        encoder_parts = [self.encoder_init, self.layer1, self.layer2, self.layer3, self.layer4]
        for part in encoder_parts:
            for param in part.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Descente (Encodeur)
        skip1 = self.encoder_init(x)         # (B, 64, H/2, W/2)
        x_pool = self.maxpool(skip1)         # (B, 64, H/4, W/4)
        skip2 = self.layer1(x_pool)          # (B, 64, H/4, W/4)
        skip3 = self.layer2(skip2)           # (B, 128, H/8, W/8)
        skip4 = self.layer3(skip3)           # (B, 256, H/16, W/16)
        bottleneck = self.layer4(skip4)      # (B, 512, H/32, W/32)

        # Remontee (Decodeur)
        d4 = self.up4(bottleneck, skip4)     # (B, 256, H/16, W/16)
        d3 = self.up3(d4, skip3)             # (B, 128, H/8, W/8)
        d2 = self.up2(d3, skip2)             # (B, 64, H/4, W/4)
        d1 = self.up1(d2, skip1)             # (B, 64, H/2, W/2)

        # Couche de sortie finale
        out = self.final_up(d1)              # (B, 2, H, W)
        return out
