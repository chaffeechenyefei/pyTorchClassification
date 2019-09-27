#Example of extracting feature using CPU where model is trained by GPU par
import models.models as models
from dataset import *
from utils import (write_event, mean_df, ThreadingDataLoader as DataLoader,
                   ON_KAGGLE)
from pathlib import Path
import torch
from aug import *
from torch import nn, cuda
import cv2

from collections import OrderedDict
from typing import Dict

def load_par_gpu_model_cpu_v2(model: nn.Module, path: Path) -> Dict:
    device = torch.device('cpu')
    state = torch.load(str(path),map_location=device)
    cur_state = model.state_dict()
    # create new OrderedDict that does not contain `module.`
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[7:]  # remove `module.`
        if str(name).startswith(('net.','vlad.')):
            new_state[name] = v
    # load params
    cur_state.update(new_state)
    model.load_state_dict(cur_state)
    # model.load_state_dict(new_state)
    # model.load_state_dict(state['model'])
    return state


def torch_load_image(patch):
    HWsize = 256
    image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image ,(HWsize, HWsize))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, HWsize, HWsize])
    image = image / 255.0
    return torch.FloatTensor(image)


model_root = '/Users/yefeichen/Desktop/Work/Downloads/'
ckpt = 'max_auc_model_954.pth'
model_name = 'vggnetvlad'

data_root = '/Users/yefeichen/Desktop/Work/Project/PanoramaImageViewer/ww_sample/'

print(data_root)
# N_Cls = 109
N_Cls = 253

model_root = Path(model_root)

# model = getattr(models, model_name)(
#             num_classes=N_Cls, pretrained='imagenet')

model = models.vggnetvlad()
# model = torch.nn.DataParallel(model)

use_cuda = cuda.is_available()

if use_cuda:
    model = model.cuda()

model_path = os.path.join(model_root,ckpt)
load_par_gpu_model_cpu_v2(model,model_path)


print('weights loaded!')
# targets = torch.rand(N_Cls,1)*10
# targets = targets.int()%10

fcnt = 0
ecnt = 0

fileList = os.listdir(data_root)

for _file in fileList:
    _portion = os.path.splitext(_file)
    if _portion[1] in ['.jpg','.png','.jpeg']:
        imgName = _file
        imgPath = os.path.join(data_root,imgName)
        img = cv2.imread(imgPath)
        try:
            img.shape
        except:
            print('Err:' + imgName)
            ecnt += 1
            continue

        inputs = torch_load_image(img)
        if use_cuda:
            inputs = inputs.cuda()

        inputs = torch.unsqueeze(inputs,0)

        with torch.no_grad():
            feat = model(inputs)
            feat = feat.squeeze()

        feat = feat.data.cpu().numpy()

        fcnt += 1
        if fcnt % 100 == 0:
            print('Num ' + str(fcnt) + ' processing...')
        npName = _portion[0] + '.npy'
        npPath = os.path.join(data_root,npName)
        np.save(npPath, feat)


