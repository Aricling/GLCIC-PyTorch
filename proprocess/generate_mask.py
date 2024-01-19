# generate mask on the image based on the point list
# input: image, point list
# output: masked image
import cv2
import numpy as np
import os

# def generate_mask(image, point_list):
#     # generate mask
#     mask = np.zeros(image.shape, dtype=np.uint8)
#     mask = cv2.fillPoly(mask, [np.array(point_list)], (255, 255, 255))
#     # cv2.imshow('mask', mask)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return mask

# def generate_img_crop(image, point_list):
#     # generate mask
#     mask = generate_mask(image, point_list)
#     # crop image
#     img_crop = cv2.bitwise_and(image, mask)
#     # cv2.imshow('img_crop', img_crop)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return img_crop


# draw a wthite circle on the image based on x0 and y0
# input: image, x0, y0
# output: image with a white circle on it
def draw_circle(image, x0, y0):
    cv2.circle(image, (x0, y0), 20, (255, 255, 255), -1) # -1 means filled circle， 20 means radius not diameter
    return image

if __name__=='__main__':
    tracked_lms_dict={
        "Avatar1_pro_20231026" : [503, 677, 566, 708, 636, 731, 708, 709, 748, 690]
    }
    x0,y0,x1,y1,x2,y2,x3,y3,x4,y4 = tracked_lms_dict["Avatar1_pro_20231026"]
    point_list = [(x0,y0),(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    # img=draw_circle(cv2.imread('videos_raw/00028.jpg'), x0, y0)
    # img=draw_circle(img, x4, y4)

    img=cv2.imread('videos_raw/00028.jpg')
    # crop image with a square
    img_crop=img[557:557+320,464:464+320,...] 
    cv2.imwrite('images/00028_crop.jpg', img_crop)


    cv2.imshow('image', img_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()