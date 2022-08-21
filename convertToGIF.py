import argparse
from PIL import Image
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True, help='to-be-convert image folder')
parser.add_argument('--gif_name', type=str, required=True, help='target gif path')
opt = parser.parse_args()


# collect frames (images)
# apply natural sorting to file name (epoch_xx.png)
image_filse = sorted(os.listdir(opt.image_dir))

def atoi(text):
    t = int(text) if text.isdigit() else text
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

image_filse.sort(key=natural_keys)

frames = list()
for image_file in image_filse:
    frames.append(Image.open(f"{opt.image_dir}/{image_file}"))

 
# save into a gif 
# duration means how many ms to be displayed of each frame
frames[0].save(
            opt.gif_name, 
            format='GIF',
            append_images=frames[1:], 
            save_all=True, 
            duration=200,
            loop=0)