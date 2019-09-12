import torch
import torch.nn as nn
import torch.nn.functional as F
##Alibaba Search Image

CUDA_FLAG = torch.cuda.is_available()

class maskLayer(nn.Module):
    def __init__(self):
        super(maskLayer, self).__init__()
        self.k = 100.0

    def forward(self, input, imgSize):
        # input: N,4

        x = torch.mm(input, torch.transpose(torch.tensor([1.0, 0, 0, 0]).unsqueeze(0).cuda(), 1, 0)) * imgSize  # N,1
        y = torch.mm(input, torch.transpose(torch.tensor([0, 1.0, 0, 0]).unsqueeze(0).cuda(), 1, 0)) * imgSize  # N,1
        xr = torch.mm(input, torch.transpose(torch.tensor([1.0, 0, 1.0, 0]).unsqueeze(0).cuda(), 1, 0)) * imgSize  # N,1
        yb = torch.mm(input, torch.transpose(torch.tensor([0, 1.0, 0, 1.0]).unsqueeze(0).cuda(), 1, 0)) * imgSize  # N,1
        # xr = x + w
        # yb = y + h
        xcols = torch.arange(0, imgSize, dtype=torch.float)  # 1,cols
        yrows = torch.arange(0, imgSize, dtype=torch.float)  # 1,rows

        if CUDA_FLAG:
            xcols = xcols.cuda()
            yrows = yrows.cuda()

        bxcols = xcols.repeat(x.size()[0], 1)  # N,cols
        byrows = yrows.repeat(x.size()[0], 1)  # N,rows

        bx = x.repeat(1, imgSize)  # N,cols
        by = y.repeat(1, imgSize)

        bxr = xr.repeat(1, imgSize)
        byb = yb.repeat(1, imgSize)

        # h(x-xl) => N,cols/rows
        hx_xl = torch.sigmoid((bxcols - bx) * self.k)
        hx_xr = torch.sigmoid((bxcols - bxr) * self.k)
        hy_yt = torch.sigmoid((byrows - by) * self.k)
        hy_yb = torch.sigmoid((byrows - byb) * self.k)

        s1 = (hx_xl - hx_xr).unsqueeze(1).repeat(1, imgSize, 1)  # N,cols -> N,1,cols -> N,rows,cols
        s2 = (hy_yt - hy_yb).unsqueeze(2).repeat(1, 1, imgSize)  # N,rows -> N,rows,1 -> N,rows,cols

        s3 = (s1 * s2).unsqueeze(1)  # N,1,rows,cols

        return s3


class gaussianMaskLayer(nn.Module):
    def __init__(self):
        super(gaussianMaskLayer,self).__init__()
        self._freeze_center = False
        self._freeze_radius = False


    def forward(self,input, imgSize):
        #input = [N,4] cx,cy,deltax,deltay
        N = input.shape[0]

        if self._freeze_radius:
            deltax = torch.ones(N,1)*0.5*imgSize
            deltay = torch.ones(N,1)*0.5*imgSize
        else:
            deltax, deltay = input[:,2]*imgSize,input[:,3]*imgSize

        if self._freeze_center:
            cx = torch.ones(N,1)*0.5*imgSize
            cy = torch.ones(N,1)*0.5*imgSize
        else:
            cx,cy = input[:,0]*imgSize,input[:,1]*imgSize #[N,1]

        xcols = torch.arange(0, imgSize, dtype=torch.float).view(1,-1)  # 1,cols
        yrows = torch.arange(0, imgSize, dtype=torch.float).view(1,-1)  # 1,rows

        if CUDA_FLAG:
            xcols = xcols.cuda()
            yrows = yrows.cuda()
            cx = cx.cuda()
            cy = cy.cuda()
            deltax = deltax.cuda()
            deltay = deltay.cuda()

        bxcols = xcols.unsqueeze(1).expand(N,imgSize,-1)
        byrows = yrows.unsqueeze(2).expand(N,-1,imgSize) #[N,H,W]


        bx = cx.view(-1,1).unsqueeze(2).expand(-1, imgSize,imgSize)  # N,cols,rows
        by = cy.view(-1,1).unsqueeze(2).expand(-1, imgSize,imgSize)

        bdeltax = deltax.view(-1,1).unsqueeze(2).expand(-1, imgSize,imgSize)
        bdeltay = deltay.view(-1,1).unsqueeze(2).expand(-1, imgSize,imgSize)

        B = (bxcols - bx)**2 / (2*bdeltax**2) + (byrows - by)**2 / (2*bdeltay**2)
        A = torch.exp(-B)
        A = A.unsqueeze(1)

        return A

    def freeze_center(self):
        self._freeze_center = True

    def freeze_radius(self):
        self._freeze_radius = True





if __name__ == '__main__':
    batchsize = 64
    k = 4
    embeddings = torch.rand((batchsize,k))
    embeddings.requires_grad = True
    # print(embeddings)
    M = maskLayer()
    feat = M.forward(embeddings,32)

    print(feat[0,0,:,:].squeeze())

    print('Done')



