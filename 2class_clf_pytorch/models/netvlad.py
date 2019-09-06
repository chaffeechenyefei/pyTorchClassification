import torch
from torch import nn
# from torch.nn import functional as F



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


if __name__ == '__main__':
    input_channels = 3
    K = 6
    inputs = torch.randn(64,input_channels,5,5)
    net = NetVladLayer(3,K)

    vlad = net(inputs)