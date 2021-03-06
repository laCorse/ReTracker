from __future__ import print_function, absolute_import
import os
import os.path as osp
from PIL import Image
import numpy

import torch
from torch.utils.data import Dataset,DataLoader



def read_imge(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("Image path {} is not exits !".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_imge(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


# if __name__ == '__main__':
#     import data_manager
#     dataset = data_manager.Market1501(root = '/home/qianchen/reid/data')
#     train_loader = ImageDataset(dataset.train)
#     from IPython import embed
#     embed()