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


from preprocess.generate_mask import get_mask_new, define_crop_box

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
        save_image(imgs, f"concatted/{args.output_img}", nrow=3)
        inpainted_pil=transforms.ToPILImage()(inpainted[0])
        inpainted_pil.save(args.output_img)
    print('output img was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args, merge_point_list=[503, 677, 566, 708, 636, 731, 708, 709, 748, 690])
