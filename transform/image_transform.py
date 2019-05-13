import numpy as np
import random
from PIL import Image, ImageFilter
import skimage


class RandomBlur(object):
    def __init__(self, prob=0.5, radius=2):
        self.prob = prob
        self.radius = radius

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        radius = random.uniform(0, self.radius)
        filter = [
            ImageFilter.GaussianBlur(radius),
            ImageFilter.BoxBlur(radius),
            ImageFilter.MedianFilter(size=3)
        ]
        img = img.filter(random.choice(filter))
        return img


class RandomNoise(object):
    def __init__(self, prob=0.5, noise=0.01):
        self.prob = prob
        self.noise = noise

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        img = np.array(img)
        mode = [
            lambda x: skimage.util.random_noise(x, 'gaussian', mean=0, var=self.noise),
            lambda x: skimage.util.random_noise(x, 'speckle', mean=0, var=self.noise),
            lambda x: skimage.util.random_noise(x, 's&p', amount=self.noise),
        ]
        img = (random.choice(mode)(img) * 255).astype(np.uint8)
        img = Image.fromarray(img)
        return img


class RandomBlackPoint(object):
    def __init__(self, prob=0.3, noise=0.001):
        self.prob = prob
        self.noise = noise
        self.kernel_size = [1, 2, 3]
        self.blur = [
            lambda x: cv2.GaussianBlur(x, (5, 5), 1.0),
            lambda x: cv2.GaussianBlur(x, (7, 7), 1.2)
        ]

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        img = np.array(img)
        mask = np.random.choice((1, 0), img.shape[:2], p=[self.noise, 1- self.noise]).astype(np.float)
        mask = cv2.dilate(mask, (random.choice(self.kernel_size), random.choice(self.kernel_size)))
        mask = random.choice(self.blur)(mask)
        mask = mask / mask.max()
        mask = 1 - mask
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)

        img = (img * mask).astype(np.uint8)
        img = Image.fromarray(img)
        return img


class ResizeWithPad(object):

    def __init__(self, size, delta_h=5, interpolation=Image.BILINEAR):
        self.size = size
        self.delta_h = delta_h
        self.interpolation = interpolation

    def __call__(self, img):
        iw, ih = img.size
        tw, th = self.size

        w = int(round(iw * th / float(ih)))
        x = random.randint(0, max(0, tw - w))

        dh = random.randint(0, self.delta_h)
        h = th - dh
        y = random.randint(0, dh)

        img_resized = img.resize((w, h), self.interpolation)
        img = Image.new('RGB', self.size, (127, 127, 127))
        img.paste(img_resized, (x, y))

        return img


class Normalize(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = transforms.functional.to_tensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomAspect(object):

    def __init__(self, aspect=(3./4., 4./3.), interpolation=Image.BILINEAR):
        self.aspect = aspect
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        aspect = random.uniform(self.aspect[0], self.aspect[1])
        ow = int(w * aspect)
        oh = int(h / aspect)
        return img.resize((ow, oh), self.interpolation)

