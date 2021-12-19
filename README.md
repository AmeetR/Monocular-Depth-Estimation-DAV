# Monocular-Depth-Estimation-DAV
An implementation of Depth Attention Volume as detailed in https://arxiv.org/pdf/2004.02760.pdf. 


## DAV Intuition 

Given two image points $P_0 = (x_0, x_1)$ and $P_1 = (x_1, y_1)$ with depth values $d_0$ and $d_1$ respectively, the depth-attention $A(P_0, P_1)$ is the ability of $P_1$ to predict the depth of $P_0$. This ability is quantied as a confience in the range of $[0,1]$, where $0$ is no ability and $1$ is maximum certainty that it is a good predictor. 