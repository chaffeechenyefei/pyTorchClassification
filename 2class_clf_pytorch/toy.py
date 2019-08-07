import torch
import torch.nn as nn
import torch.nn.functional as F


class maskLayer(nn.Module):
    def __init__(self):
        super(maskLayer, self).__init__()
        self.k = 100.0

    def forward(self, input, imgSize):
        # input: N,4

        x = torch.mm(input, torch.transpose(torch.tensor([1.0, 0, 0, 0]).unsqueeze(0), 1, 0)) * imgSize  # N,1
        y = torch.mm(input, torch.transpose(torch.tensor([0, 1.0, 0, 0]).unsqueeze(0), 1, 0)) * imgSize  # N,1
        xr = torch.mm(input, torch.transpose(torch.tensor([1.0, 0, 1.0, 0]).unsqueeze(0), 1, 0)) * imgSize  # N,1
        yb = torch.mm(input, torch.transpose(torch.tensor([0, 1.0, 0, 1.0]).unsqueeze(0), 1, 0)) * imgSize  # N,1
        # xr = x + w
        # yb = y + h
        xcols = torch.arange(0, imgSize, dtype=torch.float)  # 1,cols
        yrows = torch.arange(0, imgSize, dtype=torch.float)  # 1,rows

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

        s1 = (hx_xl - hx_xr).unsqueeze(1).repeat(1, imgSize, 1)  # N,ools -> N,1,cols -> N,rows,cols
        s2 = (hy_yt - hy_yb).unsqueeze(2).repeat(1, 1, imgSize)  # N,rows -> N,rows,1 -> N,rows,cols

        s3 = (s1 * s2).unsqueeze(1)#N,1,rows,cols

        return s3

batchsize = 64
k = 4
embeddings = torch.rand((batchsize,k))
embeddings.requires_grad = True
# print(embeddings)
M = maskLayer()

feat = M.forward(embeddings,32)

print(feat)



