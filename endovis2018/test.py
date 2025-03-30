import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import os
import time
import torch.backends.cudnn as cudnn
from PIL import Image
from model.semiseg.dpt import DPT


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('endovis2018/test_path_endovis2018.txt', 'r') as m:
        ids = m.read().splitlines()
    root_path = '' # path of endovis2018release1
    output_folder = 'endovis2018/output'
    checkpoint = torch.load('endovis2018/best_endovis2018.pth')
    
    data_transform = transforms.Compose([
        transforms.Resize((1022, 1274)), #  (1024, 1280)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
   
    cudnn.enabled = True
    cudnn.benchmark = True 

    model_configs = {'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]}}

    model = DPT(**{**model_configs['small'], 'nclass': 2})
    model.cuda()
    new_state_dict = {}
    for key in checkpoint['model_ema']:
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = checkpoint['model_ema'][key]
    model.load_state_dict(new_state_dict)

    model.eval()

    for idd in ids:
        image_pil = Image.open(os.path.join(root_path, idd.split(' ')[0])).convert('RGB')
        img = data_transform(image_pil)
        img = torch.unsqueeze(img, dim=0)

        img = img.to(device)  
        model = model.to(device)

        pred = model(img)
        pred = pred.argmax(dim=1)
        pred_visual = pred.cpu().numpy()
        pred_visual = np.squeeze(pred_visual, axis=0)
        gray_image = (pred_visual * 255).astype(np.uint8)

        output_filename = os.path.join('seq_' + idd[4] + '_' + idd.split(' ')[0][-7:-4] + '_pred.jpg')
        # print(output_filename)
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, gray_image)
        
        print(f"Saved prediction result to {output_path}")
