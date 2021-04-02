import torch
from torchvision import transforms
import cv2
from heatmaps_generation import *

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
    def __init__(self, img_id, dictionnary_labels_per_image, train_images):
        self.img_id = img_id
        self.normalize = Normalize()
        self.dictionnary_labels_per_image = dictionnary_labels_per_image
        self.train_images = train_images

    def __getitem__(self, index):
        img = cv2.resize(cv2.imread(self.train_images+'{}.jpg'.format(self.img_id[index])), (512, 512))
        img = self.normalize(img)
        img = img.transpose([2, 0, 1])
        heatmap, offset_x, offset_y, object_size_x, object_size_y = generate_heatmap_offset(self.img_id[index], 1, self.dictionnary_labels_per_image)
        regr = np.zeros((2, 128, 128))
        regr[0, :, :] = object_size_x
        regr[1, :, :] = object_size_y
        return img, self.img_id[index], heatmap, regr

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
