import torch
from torch import nn
from torch.nn import functional as F



class NetVladLayer(nn.Module):
    """
    NetVLAD: CNN architecture for weakly supervised place recognition,cvpr,2016   
    """
    def __init__(self,input_channels,centers_num):
        super(NetVladLayer,self).__init__()
        self.input_channels = input_channels    #D
        self.centers_num = centers_num  #K
        self.conv = nn.Conv2d(input_channels,centers_num,(1,1))#1x1xDxK

        self.centers = nn.Parameter(torch.Tensor(input_channels, centers_num)) #DxK
        self.centers.data.uniform_(-0.1, 0.1)

        self.output_features = self.centers_num*self.input_channels #K*D


    def forward(self, inputs):
        """
        
        :param inputs:[B,D,H,W] 
        :return: 
        """
        K = self.centers_num
        B,D,H,W = inputs.shape
        N = H*W
        X = inputs.view(B,D,1,N) #[B,D,1,N]
        LinearX = self.conv(inputs) #[B,K,H,W]

        LinearX = LinearX.view(B,K,N) #[B,K,H*W]

        expLinearX = LinearX.exp() #[B,K,N]

        Sum_expLinearX = torch.sum(expLinearX,dim=1,keepdim=True)#[B,1,N]

        alpha = torch.div(expLinearX,Sum_expLinearX) #[B,K,N]

        X = X.expand(B,D,K,N)
        center = self.centers.view(1,D,K,1).expand(B,D,K,N)

        XmC = X - center #[B,D,K,N]

        alpha = alpha.view(B,1,K,N).expand(B,D,K,N)

        vlad = torch.sum(alpha*XmC,dim=3) #[B,D,K]

        norm_frac = torch.norm(vlad,dim=1,keepdim=True) #[B,1,K]

        vlad = torch.div(vlad,norm_frac) #[B,D,K]

        vlad = vlad.view(B,D*K)

        norm_frac = torch.norm(vlad,dim=1,keepdim=True) #[B,1]

        vlad = torch.div(vlad,norm_frac) #[B,D*K]

        return vlad


class NetVladLayerV2(nn.Module):
    """
    NetVLAD layer implementation
    https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
    """

    def __init__(self, num_clusters=64, dim=128, alpha=0.1,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVladLayerV2, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


if __name__ == '__main__':
    input_channels = 3
    K = 6
    inputs = torch.randn(64,input_channels,5,5)
    net = NetVladLayer(3,K)

    vlad = net(inputs)