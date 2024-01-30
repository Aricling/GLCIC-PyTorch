import os
import argparse
import json
from tkinter import Y
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import CompletionNetwork
# from utils import poisson_blend, gen_input_mask
from utils import poisson_blend


from proprocess.generate_mask import get_mask_new

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)


def define_crop_box(merge_point_list):
    x0,y0,x1,y1,x2,y2,x3,y3,x4,y4=merge_point_list
    # width=x4-x0
    # height=max(y2-y0,y2-y4,150)
    # # crop box should be 1.2 times of width and height
    # width*=1.2
    # height*=1.2
    # x_left=int(round(1.1*x0-0.1*x4))-50
    # y_top=int(round(1.1*y0-0.1*y2))-75
    # x_right=x_left+int(round(width))+100
    # y_bottom=y_top+int(round(height))+50
    x_middle=int(round((x0+x4)/2))
    y_middle=int(round((y0+y2)/2))
    x_left=x_middle-160
    y_top=y_middle-160
    x_right=x_middle+160
    y_bottom=y_middle+160
    new_merge_point_list=[x0-x_left,y0-y_top,x1-x_left,y1-y_top,x2-x_left,y2-y_top,x3-x_left,y3-y_top,x4-x_left,y4-y_top]
    return (x_left,y_top,x_right,y_bottom), new_merge_point_list

def main(args, merge_point_list):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_img = os.path.expanduser(args.input_img)
    args.output_img = os.path.expanduser(args.output_img)

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1) # what is this? "mpv": [0.5062325495504219, 0.4255871700324652, 0.38299278586700136]
    
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    crop_box, new_merge_point_list=define_crop_box(merge_point_list)
    img = Image.open(args.input_img)
    img=img.crop(crop_box)

    x = transforms.ToTensor()(img) # normalize to [0,1]
    x = torch.unsqueeze(x, dim=0)

    mask=get_mask_new(
        shape=(1, 1, x.shape[2], x.shape[3]),
        # only use the first and last point
        point_list=new_merge_point_list,
        radius=20
    ) # torch.Size([1,1,160,160])

    # inpaint
    model.eval()
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        # input:[1,4,230,394], output:[1,3,232,396]
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        save_image(imgs, args.output_img, nrow=3)
    print('output img was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args, merge_point_list=[503, 677, 566, 708, 636, 731, 708, 709, 748, 690])
