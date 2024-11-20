import torch
import torch.nn as nn

from base.base_model import BaseModel

class AlexNet(BaseModel):
    """
    A PyTorch implementation of the paper: 
    `AlexNet <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_prob: float = 0.5,
        **kwargs
    ):
        """
        Initializes the AlexNet model
        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of classes in the dataset. Defaults to 1000.
        Attributes:
            features (torch.nn.Sequential): Feature extractor of the model.
            classifier (torch.nn.Sequential): Classifier of the model.
        """
        super(AlexNet, self).__init__()
        self.features = self._make_features(in_channels)
        self.classifier = self._make_classifier(num_classes, dropout_prob)

    def _make_features(self, in_channels):
        layers = []
        layers += self._conv_block(in_channels, 64, kernel_size=11, stride=4, padding=2)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        layers += self._conv_block(64, 192, kernel_size=5, padding=2)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        layers += self._conv_block(192, 384, kernel_size=3, padding=1)
        layers += self._conv_block(384, 256, kernel_size=3, padding=1)
        layers += self._conv_block(256, 256, kernel_size=3, padding=1)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        return nn.Sequential(*layers)

    def _conv_block(self, in_channels, out_channels, **kwargs):
        return [
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.ReLU(inplace=True)
        ]

    def _make_classifier(self, num_classes, dropout_prob=0.5):
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x