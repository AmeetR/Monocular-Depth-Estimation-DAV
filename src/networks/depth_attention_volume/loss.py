"""
An implementation of the loss function for the depth attention volume network.

https://arxiv.org/pdf/2004.02760.pdf
Loss function consists of two main components:
    - Attention Loss: minimize the error between the estimated (output of DAV predictor) and the ground truth DAV. L_mae is the meana bsolute erorr between
    the predicted and the ground truth DAV.In addition, minimize the angle between the predicted and the ground trueth depth-attention maps. 
    Full attention loss is the addition of these two things.
"""

def loss_mae(gt, pred, height, width):
    """
    Mean absolute error of the depth attention volume.
    """
    weight = 1/(height*width)
    pass

def loss_angle(gt, pred, height, width):
    """
    Angle between the predicted and the ground truth depth-attention maps.
    """
    pass

def loss_grad(gr, pred, M, alpha):
    """
    Loss which is used to penalize sudden changes of edge structures in both x and y directions
    """