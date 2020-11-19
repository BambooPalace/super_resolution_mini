from PIL import Image
import torch
from torchvision.utils import save_image
import glob
import os

def main():
    'generate 4x scale fake ground truth for using tools/test.py'

    paths = sorted(glob.glob('/Users/clairegong/Downloads/data/test/*.png'))
    dest = '/Users/clairegong/Downloads/data/fake'

    for path in paths:
        filename = path.split('/')[-1]
        #check image size
        img = Image.open(path)
        w, h = img.size
        fake = torch.rand(3, h*4, w*4)*100
        print('saving ', filename)
        save_image(fake, os.path.join(dest,filename) )

if __name__ == '__main__':
    main()