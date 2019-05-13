import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from transform import *
import cv2


class RecognizeDataset(Dataset):

    def __init__(self, root, sep='_', transform=None, target_transform=None):
        self.root = root
        self.sep = sep
        self.transform = transform
        self.target_transform = target_transform
        self.image_list = []
        for root, dirs, files in os.walk(self.root):
            for name in files:
                if name[-4:] in [".jpg", ".png"]:
                    self.image_list.append(os.path.join(root, name))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        path = self.image_list[index]
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        _, name = os.path.split(path)
        stem, ext = os.path.splitext(name)
        label = stem.split(self.sep)[-1]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class AlignCollate(object):

    def __init__(self, imgH=32, align=8, max_ratio=20):
        self.imgH = imgH
        self.align = align
        self.max_ratio = max_ratio
        
    def __call__(self, batch):
        images, labels = zip(*batch)

        ratios = []
        for image in images:
            w, h = image.size
            ratios.append(w / float(h))
        ratios.sort()
        max_ratio = min(ratios[-1], self.max_ratio)

        if isinstance(self.imgH, int):
            imgH = self.imgH
        elif isinstance(self.imgH, (list, tuple)):
            imgH = random.choice(self.imgH)
        else:
            imgH = 32
        imgW = int(np.ceil(max_ratio * imgH / self.align) * self.align)

        transform = transforms.Compose([
            ResizeWithPad((imgW, imgH), delta_h=int(imgH / 6.)),
            Normalize()
        ])
        images = [transform(image).unsqueeze(0) for image in images]
        images = torch.cat(images, 0)

        return images, labels


if __name__ == "__main__":
    import cv2
    path = "/home/gaochao/data/general_recog/train/chinese_images"
    dataset = RecognizeDataset(path)
    print(len(dataset))
    transform = transforms.Compose([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        RandomNoise(prob=0.5, noise=0.01),
        RandomBlur(prob=0.5, radius=1.5),
        RandomAspect(aspect=(3./4., 4./3.)),
        transforms.RandomAffine(3., translate=(0., 0.05), shear=2., fillcolor=(127, 127, 127)),
        ResizeWithPad((320, 32))
    ])
    for image, label in dataset:
        image = transform(image)
        cv2.imshow("", np.array(image)[:,:,::-1])
        key = cv2.waitKey(1000)
        if key == 27:
            break
