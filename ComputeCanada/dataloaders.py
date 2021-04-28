import torch
import numpy as np
from torchvision import transforms
import cv2
from heatmaps_generation import *
from PIL import Image, ImageDraw
from torchvision.transforms import ColorJitter
from torchvision.transforms import GaussianBlur

class Normalize(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.norm = transforms.Normalize(self.mean, self.std)

    def __call__(self, image):
        image = (image.astype(np.float32) / 255)
        image -= self.mean
        image /= self.std
        return image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_id, augment_data, train_images, dictionnary_labels_per_image):
        self.img_id = img_id
        self.normalize = Normalize()
        self.augment_data = augment_data
        self.train_images = train_images
        self.dictionnary_labels_per_image = dictionnary_labels_per_image

    def __getitem__(self, index):
        if self.augment_data == False:
            img = cv2.resize(cv2.imread(self.train_images+'{}.jpg'.format(self.img_id[index])), (512, 512))
        else:
            img = Image.open(self.train_images+'{}.jpg'.format(self.img_id[index])).resize((512, 512))
            color = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.4)
            blur = GaussianBlur((3, 3), sigma=(0.1, 2))
            img = color(img)
            img = blur(img)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = self.normalize(img)
        img = img.transpose([2, 0, 1])
        heatmap, offset_x, offset_y, object_size_x, object_size_y = generate_heatmap_offset(self.img_id[index], self.dictionnary_labels_per_image)
        regr = np.zeros((2, 128, 128))
        offset = np.zeros((2, 128, 128))
        regr[0, :, :] = object_size_x
        regr[1, :, :] = object_size_y

        offset[0, :, :] = offset_x
        offset[1, :, :] = offset_y
        return img, self.img_id[index], heatmap, regr, offset

    def __len__(self):
        return len(self.img_id)


class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, img_id, test_dir):
        self.img_id = img_id
        self.normalize = Normalize()
        self.test_dir = test_dir

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        img = cv2.resize(cv2.imread(self.test_dir+'{}'.format(self.img_id[index])), (512, 512))
        img = self.normalize(img)
        img = img.transpose([2, 0, 1])
        return img, self.img_id[index]
