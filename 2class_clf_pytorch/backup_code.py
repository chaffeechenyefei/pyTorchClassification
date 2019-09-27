##======================================================================================================================
##学习率设置方法储备
##======================================================================================================================
##Method::1
# conv5_params = list(map(id, net.conv5.parameters()))
# conv4_params = list(map(id, net.conv4.parameters()))
# base_params = filter(lambda p: id(p) not in conv5_params + conv4_params,
#                      net.parameters())
# optimizer = torch.optim.SGD([
#             {'params': base_params},
#             {'params': net.conv5.parameters(), 'lr': lr * 100},
#             {'params': net.conv4.parameters(), 'lr': lr * 100},
#             , lr=lr, momentum=0.9)
##Method::2
# model = Net()
# conv_params = list(map(id, model.conv1.parameters()))  # 提出前两个卷积层存放参数的地址
# conv_params += list(map(id, model.conv2.parameters()))
# prelu_params = []
# for m in model.modules():  # 找到Prelu的参数
#     if isinstance(m, nn.PReLU):
#         prelu_params += m.parameters()
#
# # 假象网络比我写的很大，还有一部分参数，这部分参数使用另一个学习率
# rest_params = filter(lambda x: id(x) not in conv_params + list(map(id, prelu_params)), model.parameters())  # 提出剩下的参数
# print(list(rest_params))
# '''
# >> []   #是空的，因为我举的例子没其他参数了
# '''
# import torch.optim as optim
#
# optimizer = optim.Adam([{'params': model.conv1.parameters(), 'lr': 0.2},
#                         {'params': model.conv2.parameters(), 'lr': 0.2},
#                         {'params': prelu_params, 'lr': 0.02},
#                         {'params': rest_params, 'lr': 0.3}




##======================================================================================================================
##solve the problem of path root of package missing like 'No modules named dataset is found'
##======================================================================================================================
#solve the problem of path root of package missing like 'No modules named dataset is found'
# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)