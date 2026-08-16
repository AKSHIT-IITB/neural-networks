"""CNN for CIFAR-10: 3 conv blocks -> classifier head."""
import torch.nn as nn


class Cifar10CNN(nn.Module):
    """
    Input 3x32x32
      Conv(3->32)  + BN + ReLU, Conv(32->32)  + BN + ReLU, MaxPool -> 32x16x16
      Conv(32->64) + BN + ReLU, Conv(64->64)  + BN + ReLU, MaxPool -> 64x8x8
      Conv(64->128)+ BN + ReLU, Conv(128->128)+ BN + ReLU, MaxPool -> 128x4x4
      Flatten -> Dropout -> Linear(2048->256) -> ReLU -> Dropout -> Linear(256->10)
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
