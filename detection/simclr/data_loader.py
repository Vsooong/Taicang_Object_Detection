import os
from PIL import Image
from torchvision import transforms
from detection.simclr.simclr import TransformsSimCLR
import numpy as np

image_path = 'D:/samples'
image_save_path = 'D:/samples'


def similar_picture(image_path):
    imgs = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    transform = TransformsSimCLR()
    for path in imgs:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        img = np.asarray(img)
        im = Image.fromarray(img)
        save_pth = os.path.join(image_save_path, os.path.basename(path).replace('.jpg', '-sim.jpg'))
        im.save(save_pth)
        print('Image saves to:', save_pth)
        im.close()


def clear_test_samples(test_img_save_path):
    for i in sorted(os.listdir(test_img_save_path)):
        if i.endswith('-sim.jpg'):
            path = os.path.join(test_img_save_path, i)
            os.remove(path)
            print('removed: ', path)


similar_picture(image_path)
# clear_test_samples(image_save_path)
