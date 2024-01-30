# generate mask on the image based on the point list
# input: image, point list
# output: masked image
import cv2
import numpy as np
import os
import torch

# def generate_mask(image, point_list):
#     # generate mask
#     mask = np.zeros(image.shape, dtype=np.uint8)
#     mask = cv2.fillPoly(mask, [np.array(point_list)], (255, 255, 255))
#     # cv2.imshow('mask', mask)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return mask

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

def get_mask_new(
        shape,
        point_list,
        radius=20
):
    mask = torch.zeros(shape)
    x0,y0,x1,y1,x2,y2,x3,y3,x4,y4=point_list
    # print(point_list)
    # make values around (x0,y0) in the mask to be 1
    mask[0,0,y0-radius:y0+radius,x0-radius:x0+radius]=1.0
    # make values around (x4,y4) in the mask to be 1
    mask[0,0,y4-radius:y4+radius,x4-radius:x4+radius]=1.0

    return mask


if __name__=='__main__':
    pass
   