import os
import random
import json
import numpy as np
from torch.utils import data
from PIL import Image
import pse_utils as pse


def load_icdar(anno_file, dontcare=["###", "-", "/"]):
    with open(anno_file) as f:
        data = [line.strip().strip("\ufeff").strip('\xef\xbb\xbf').split(",") for line in f.readlines()]
        if dontcare is not None and len(dontcare) > 0:
            data = filter(lambda x: x[8] not in dontcare, data)
        gt = [list(map(float, items[:8])) for items in data]
        return gt


def load_json(anno_file, dontcare=None):
    with open(anno_file) as f:
        data = json.load(f)["objects"]
        if dontcare is not None and len(dontcare) > 0:
            data = filter(lambda x: x["label"] not in dontcare, data)
        gt = [list(np.array(items["polygon"]).reshape(-1)) for items in data]
        return gt


class DataLoader(object):
    def __init__(self, image_root, annotation_root, prefix, postfix, loader):
        self.image_root = image_root
        self.annotation_root = annotation_root
        self.prefix = prefix
        self.postfix = postfix
        self.loader = loader

    def __call__(self):
        data = []
        for name in os.listdir(self.image_root):
            stem, ext = os.path.splitext(name)
            image_path = os.path.join(self.image_root, name)
            anno_path = os.path.join(self.annotation_root, self.prefix + stem + self.postfix)
            if not os.path.exists(anno_path):
                continue
            image = Image.open(image_path)
            gt_boxes = np.array(self.loader(anno_path))
            size = np.array(image.size)
            data.append({"image": name, "size": size, "gt_boxes": gt_boxes})
        return data


class MaskGenerator(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
    
    def __call__(self, data):
        out = pse.GenerateLabel(np.round(data["gt_boxes"]).astype(np.int32), data["size"][0], data["size"][1], self.n, self.m)
        out = np.concatenate([np.expand_dims(o, 2) for o in out], axis=2)
        out = Image.fromarray(out)
        return out


class TextData(data.Dataset):
    def __init__(self, image_root, annotation_root, prefix="gt_", postfix=".txt", loader=load_icdar,
                 joint_transform=None, input_transform=None, target_transform=None):
        self.image_root = image_root
        self.annotation_root = annotation_root
        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        self.data_loader = DataLoader(self.image_root, self.annotation_root,
                                      prefix=prefix, postfix=postfix, loader=loader)

        self.data = self.data_loader()
        self.mask_generator = MaskGenerator(3, 0.6)
        self.shuffle()

    def __getitem__(self, index):
        item = self.data[index]
        size = item["size"]
        image = Image.open(os.path.join(self.image_root, item["image"]))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        mask = self.mask_generator(item)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
    
    def __len__(self):
        return len(self.data)

    def shuffle(self):
        random.shuffle(self.data)


if __name__ == '__main__':
    import cv2
    dataset = TextData("./data_pdf/train/image", "./data_pdf/train/txt")
    for image, mask in dataset:
        cv2.imshow("", np.array(mask))
        key = cv2.waitKey()
        if key == 27:
            exit(0)
