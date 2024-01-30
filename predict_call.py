import os
from tqdm import tqdm

if os.path.exists('output_images'):
    pass
else:
    os.mkdir('output_images')

img_name_list=sorted(os.listdir('videos_raw'), key=lambda x: int(x.split('.')[0]))
for img_name in tqdm(img_name_list, desc='predicting', total=len(img_name_list)):
    os.system(f'python \
              predict.py \
              model_cn config.json \
              videos_raw/{img_name} \
              output_images/{img_name}')