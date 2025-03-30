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
import sys
from model.semiseg.dpt import DPT

def list_jpg_files(folder_path):
    jpg_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_path = 'spine/input'
    output_folder = 'spine/output'
    checkpoint = torch.load('spine/best_spine.pth')
    
    jpg_files = list_jpg_files(folder_path)
    data_transform = transforms.Compose([
        transforms.Resize((728, 728)),
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

    for file in jpg_files:
        image_init = cv2.imread(os.path.join(folder_path, file))
        image_pil = Image.fromarray(image_init.astype('uint8')).convert('RGB')

        img = data_transform(image_pil)
        img = torch.unsqueeze(img, dim=0)

        img = img.to(device)  
        model = model.to(device)

        pred = model(img)
        pred = pred.argmax(dim=1)
        pred_visual = pred.cpu().numpy()
        pred_visual = np.squeeze(pred_visual, axis=0)
        gray_image = (pred_visual * 255).astype(np.uint8)

        output_filename = os.path.splitext(os.path.basename(file))[0] + '_pred.jpg'
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, gray_image)
        
        print(f"Saved prediction result to {output_path}")
        