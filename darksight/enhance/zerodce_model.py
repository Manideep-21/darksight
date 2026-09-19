"""Zero-DCE network, vendored from https://github.com/Li-Chongyi/Zero-DCE (Zero-DCE_code/model.py).

Guo, Li, Guo, Loy, Hou, Kwong, Cong. "Zero-Reference Deep Curve Estimation for
Low-Light Image Enhancement." CVPR 2020. Original code: academic research use only,
CC BY-NC 4.0. Changes: removed unused pooling layers and the dead imports; the layer
names are unchanged so the official Epoch99.pth state_dict loads as-is.
"""

import torch
import torch.nn as nn

NUM_ITERATIONS = 8


class ZeroDCENet(nn.Module):
    """DCE-Net: 7 conv layers (32 ch) with symmetric skips -> 24 curve maps (8 iterations x RGB)."""

    def __init__(self, number_f: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 3 * NUM_ITERATIONS, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor):
        """x: Nx3xHxW in [0, 1]. Returns (image after 4 iterations, final image, curve maps)."""
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        curves = torch.split(x_r, 3, dim=1)

        # Light-enhancement curve LE(x) = x + a * x * (1 - x); the original code writes it as
        # x + r * (x^2 - x), i.e. a = -r. Kept verbatim for weight compatibility.
        enhance_image_1 = x
        for i, r in enumerate(curves):
            x = x + r * (torch.pow(x, 2) - x)
            if i == 3:
                enhance_image_1 = x
        return enhance_image_1, x, x_r


# Name used in the official repository, kept as an alias.
enhance_net_nopool = ZeroDCENet
