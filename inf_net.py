import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

from load_semi import Inf_Net
from load_unet import Inf_Net_UNet

def init_semi():
    semi = Inf_Net()
    device = torch.device('cpu')
    semi.load_state_dict(torch.load(seminet_pth, map_location=device))
    semi.eval()
    return semi

def predict_semi(k):
    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = semi(k)
    res = lateral_map_2
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

def init_unet():
    infnet = Inf_Net_UNet(6,3)
    device = torch.device('cpu')
    infnet.load_state_dict(torch.load(path, map_location=device))
    infnet.eval()
    return infnet

def predict_unet(k,d):
    s = torch.cat((k,d), dim=1)
    res = infnet(s)
    res = torch.sigmoid(res) 
    b, _, w, h = res.size()
    # output b*n_class*h*w -- > b*h*w
    pred = res.cpu().permute(0, 2, 3, 1).contiguous().view(-1, 3).max(1)[1].view(b, w, h).numpy().squeeze()
    return pred

def split_class_imgs(pred):
    im_array_red = np.array(pred)  # 0, 38
    im_array_green = np.array(pred)  # 0, 75
    print(np.unique(im_array_red)) # mask value is max of this

    im_array_red[im_array_red != 0] = 1
    im_array_red[im_array_red == 0] = 255
    im_array_red[im_array_red == 1] = 0

    im_array_green[im_array_green != 1] = 0
    im_array_green[im_array_green == 1] = 255
    return im_array_green, im_array_red

def get_label(patient_number):
    with open(LABEL_PATH) as file:
        for line in file.readlines():
            if str(patient_number) in line:
                label = line.split(",")[1].rsplit("\n")[0]
                return label


