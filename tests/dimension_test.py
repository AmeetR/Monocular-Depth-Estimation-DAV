import torch
from data.DIODE_data_loader import DIODEDataset
from networks.depth_attention_volume.encoder import drn_d_22
import numpy as np

def test_encoder_dims_1():
    """
    Need to check if encoder dims are 8x downsampled image
    """
    encoder = drn_d_22()
    encoder.eval()
    # make random n x n x 3 image
    img = torch.randn(1, 3, 224, 224)
    # get output
    output = encoder(img)
    print(output.shape)
    assert output.shape[2] == 224/8
    assert output.shape[3] == 224/8


def test_encoder_dims_2():
    """
    Need to check if encoder dims are 8x downsampled image
    """
    encoder = drn_d_22()
    encoder.eval()
    # make random n x n x 3 image
    img = torch.randn(1, 3, 720, 540)
    # get output
    output = encoder(img)
    print(output.shape)
    assert output.shape[2] == 720/8
    assert output.shape[3] == int(np.round(540/8))

