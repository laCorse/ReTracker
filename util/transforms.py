from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import numpy as np

class Random2DTranslation(object):
    def __init__(self, height, width, p = 0.5, interpolation = Image.BILINEAR):
        """
        0.5的概率直接缩放到想要的长和宽,0.5的概率先进行缩放到想要的长和宽还要大一些(例如扩大到想要的1.125),再进行随机剪裁到想要的大小
        :param height:
        :param width:
        :param p:
        :param interpolation:差值方法
        """
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):

        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width,new_height),self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))

        croped_img = resized_img.crop((x1, y1, x1+self.width, y1+self.height))

        return croped_img

# test for transform
# if __name__ == '__main__':
#     img = Image.open('/home/qianchen/reid/data/market1501/bounding_box_train/0002_c1s1_000451_03.jpg')
#     transform = Random2DTranslation(256, 128, 0.5)
#     img_t = transform(img)
#     import matplotlib.pyplot as plt
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.subplot(122)
#     plt.imshow(img_t)
#     plt.show()