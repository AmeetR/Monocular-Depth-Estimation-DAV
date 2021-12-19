"""
====
Implementation of the depth attention volume. 

https://arxiv.org/pdf/2004.02760.pdf
====
We introduce a concept of depth-attention volume (DAV) to aggregate spatial information non-locally from those coplanar
structures. We use both fronto-parallel and non-fronto-parallel constraints to
learn the DAV in an end-to-end manner

Depth-attention volume (DAV) is a collection of depth-attention maps (Eq. 3,
Figure 1) obtained using each image location as a query point at a time. Therefore,
the DAV for an image of size 8H × 8W is a 4D tensor of size H × W × H × W.

The non-local depth attention module is the central component of our network with the detailed structure provided in Table 4. In that, green, blue,
orange indicate the green, blue, and orange embedding spaces mentioned in the
26 L. Huynh et al.
manuscript. “J” denotes element-wise multiplication, “L” indicates elementwise sum, and “N” is the outer product. Layers denoted with ‡‡ imply reshaping
and permuting the tensor to match the required shape for operation. Note that
green-1bn, green-1γ, green-1β and blue-1bn, blue-1γ, blue-1β are generated at
the same time as indicated by the dashed line.

The non-local depth-attention module is located between the encoder and
the decoder. It maps the input features X to the output features Y of the same
size. The primary purpose of the module is to add the non-local information
embedded in the depth-attention volume (DAV) to Y, but it is also used to
predict and learn the DAV based on the ground-truth data. The structure of the
module is presented in Figure 4.
We implement the DAV-predictor by first transforming X into green and blue
embeddings using 1×1 convolution. We exploit the symmetry of DAV, and maximize the correlation between these two spaces by applying cross-denormalization
on both green and blue embeddings. Cross-denormalization is a conditional normalization technique [4] that is used to learn an affine transformation from the
data. Specifically, the green embedding is first normalized to zero mean and unit
standard deviation using batch-normalization (BN). Then, the blue embedding
is convolved to create two tensors that are multiplied and added the normalized
features from the green branch, and vise versa. The denormalized representations
are then activated with ReLUs and transformed by another 1 × 1 convolution
before multiplying with each others. Finally, the DAV is activated using the sigmoid function to ensure that the output values are in range [0, 1]. We empirically
verified that applying cross-modulation in two embedding spaces is superior than
using a single embedding with double the number of features.
Furthermore, X is fed into the orange branch and multiplied with the estimated DAV to amplify the effect of the input features. Finally, we add a residual
connection (red) to prevent the vanishing gradient problem when training our
network.

Table 4. Internal structure of the non-local depth attention module.
Non-local depth attention module
Input Operations k s d CH RES Output
layer8-X conv 1 1 1 256 29 × 38 orange
layer8-X conv 1 1 1 1024 29 × 38 green-1
green-1 bn - - - 1024 29 × 38 green-1bn
green-1 conv 1 1 1 1024 29 × 38 green-1γ
green-1 conv 1 1 1 1024 29 × 38 green-1β
green-1bn, blue-1γ
J - - - 1024 29 × 38 green-1bn-γ
green-1bn-γ, blue-1β
L - - - 1024 29 × 38 green-1-denorm
green-1-denorm relu+conv 1 1 1 1024 29 × 38 green-2
layer8-X conv 1 1 1 1024 29 × 38 blue-1
blue-1 bn - - - 1024 29 × 38 blue-1bn
blue-1 conv 1 1 1 1024 29 × 38 blue-1γ
blue-1 conv 1 1 1 1024 29 × 38 blue-1β
blue-1bn, green-1γ
J - - - 1024 29 × 38 blue-1bn-γ
blue-1bn-γ, green-1β
L - - - 1024 29 × 38 blue-1-denorm
blue-1-denorm relu+conv 1 1 1 1024 29 × 38 blue-2
green-2‡‡, blue-2‡‡ N - - - 1 1102 × 1102 dav-1
dav-1 sigmoid - - - 1 1102 × 1102 dav-2
dav-2, orange N - - - 256 29 × 38 dav-3‡‡
dav-3 conv+bn 1 1 1 512 29 × 38 dav-4
dav-4, layer8-X L - - - 512 29 × 38 layer8-Y


"""

import torch.nn as nn
import torch

class DepthAttention(nn.Module):
    def __init__(self):
        super(DepthAttention, self).__init__()
        self.orange_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.green1_conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.green1_bn = nn.BatchNorm2d(num_features=1024)
        self.green1_gamma = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.green1_beta = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.green2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        )

        self.blue1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.blue1_bn = nn.BatchNorm2d(num_features=1024)
        self.blue1_gamma = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.blue1_beta = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.blue2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        )
        self.dav4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512)
        )

    def forward(self, x):
        # x: B x 512 x H x W
        
        orange = self.orange_conv(x)
        green_1 = self.green1_conv(x)
        green1_bn = self.green1_bn(green_1)
        green1_gamma = self.green1_gamma(green_1)
        green1_beta = self.green1_beta(green_1)
        
        blue_1 = self.blue1(x)
        blue_1_bn = self.blue1_bn(blue_1)
        blue1_gamma = self.blue1_gamma(blue_1)
        blue1_beta = self.blue1_beta(blue_1)

        green_1bn_gamma = green1_bn * blue1_gamma
        green_1_denorm = green_1bn_gamma + blue1_beta
        green2 = self.green2(green_1_denorm)

        blue_1bn_gamma = blue_1_bn * green1_gamma
        blue_1_denorm = blue_1bn_gamma + green1_beta
        blue2 = self.blue2(blue_1_denorm)
        # torch.einsum is used to calculate the outer product with batches, as torch.outer doesn't broadcast
        dav_1 = torch.einsum('bi,bj->bij', (green2, blue2)) # will need to be reshaped for this operation
        dav_2 = torch.sigmoid(dav_1)
        dav3 = torch.einsum('bi,bj->bij', (dav_2, orange))
        dav_4 = self.dav4(dav3)
        out = dav_4 + x # out should be a 4D Tensor of shape B x H x W x H x W

        return out

