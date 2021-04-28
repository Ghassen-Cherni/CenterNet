import torch
import numpy as np
from skimage import io
import os
from skimage import transform as skim_transform
import json

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Normalize(object):
    def __init__(self):
        self.mean = (0.4457, 0.4205, 0.3884)
        self.std = (0.2694, 0.2664, 0.2790)

    def __call__(self, image):
        image -= self.mean
        image /= self.std
        return image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, mode, processed_annotations=None):
        """
        path_to_data: path to the directory that holds folders "VOCtest_06-Nov-2007" and "VOCtrainval_06-Nov-2007"
        mode: 'train', 'val' or 'test'
        transform: set to True if you want to apply specific transformations
        """
        CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

        if mode == "train" or mode == 'val':
            image_set_dir = os.path.join(path_to_data, "VOCtrainval_06-Nov-2007")
        elif mode == "test":
            image_set_dir = os.path.join(path_to_data, "VOCtest_06-Nov-2007")
        else:
            image_set_dir = path_to_data
            raise ValueError("Not a valid mode")

        voc_folder = os.path.join(image_set_dir, "VOCdevkit", "VOC2007")
        image_set_dir = os.path.join(image_set_dir, "VOCdevkit", "VOC2007", "ImageSets", "Main")
        mode_file = os.path.join(image_set_dir, mode + '.txt')

        if not os.path.isfile(mode_file):
            raise ValueError("No image set found")

        # Attributes concerning the dataset's architecture
        self.jpegs = os.path.join(voc_folder, "JPEGImages")
        self.annotations = os.path.join(voc_folder, "Annotations")
        self.mode_file = mode_file
        with open(self.mode_file, 'r') as mf:
            content = mf.readlines()
        content = [x.strip() for x in content]
        self.content = content

        # Attributes concerning the data
        self.classes = CLASSES
        self.num_class = len(self.classes)
        self.index_map = dict(zip(self.classes, range(self.num_class)))

        if processed_annotations is not None:
            ann = json.load(open(processed_annotations))
            temp = {}
            for k in ann.keys():
                temp[int(k)] = ann[k]
            self.processed_annotations = temp
        else:
            self.processed_annotations = None
        self.normalize = Normalize()

    def __getitem__(self, idx):
        # print(idx)
        name = self.content[idx]
        item = int(name)
        # Get image
        image = os.path.join(self.jpegs, name + '.jpg')
        if not os.path.isfile(image):
            print(image)
            raise ValueError("No associated image")
        image = io.imread(image)
        # Get parsed labels
        if self.processed_annotations is None:
            # image = io.imread(image)
            # Get labels
            annotations_xml = os.path.join(self.annotations, name + '.xml')
            label = self.xml_parser(item, annotations_xml)
        else:
            label = self.processed_annotations[item]

        im, hm, offset, object_size = generate_heatmaps(image, label)
        im = self.normalize(im)
        im = im.transpose([2, 0, 1])
        return im, item, hm, object_size, offset

    def __len__(self):
        return len(self.content)

    def process_dataset(self, save_name=None):
        with open(self.mode_file, 'r') as mf:
            content = mf.readlines()
        image_names = [x.strip() for x in content]

        processed_annotations = {}
        for i, name in enumerate(image_names):
            idx = int(name)
            annotations_xml = os.path.join(self.annotations, name + '.xml')
            label = self.xml_parser(idx, annotations_xml)
            image = os.path.join(self.jpegs, name + '.jpg')
            if not os.path.isfile(image):
                print(image)
                raise ValueError("No associated image")
            processed_annotations[idx] = label
        self.processed_annotations = processed_annotations
        if save_name is not None:
            with open(save_name, 'w') as f:
                json.dump(processed_annotations, f)

        return processed_annotations

    def xml_parser(self, idx, idx_xml):
        root = ET.parse(idx_xml).getroot()

        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            label.append([idx, xmin, ymin, xmax, ymax, cls_id, difficult])

        # return np.array(label)
        return label


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def generate_heatmaps(image, label):
    """
    Resizes image and generates heatmaps
    """
    hm_size = 128
    heatmap = np.zeros((hm_size, hm_size))
    offset_x = np.zeros((hm_size, hm_size))
    offset_y = np.zeros((hm_size, hm_size))
    object_size_x = np.zeros((hm_size, hm_size))
    object_size_y = np.zeros((hm_size, hm_size))

    h_init, w_init, _ = image.shape
    w_scale = 64 / w_init  # i.e. 128/2
    h_scale = 64 / h_init

    im_resized = skim_transform.resize(image, (512, 512))

    for i, box in enumerate(label):
        _, x_min, y_min, x_max, y_max, _, _ = box
        size_x = float((x_max - x_min + 1))
        size_y = float((y_max - y_min + 1))
        radius = gaussian_radius((size_y, size_x), 0.3)
        radius = max(0, int(radius))
        sigma_x = float(radius / 3)
        sigma_y = float(radius / 3)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_x_resized_float = center_x / w_init * hm_size
        center_y_resized_float = center_y / h_init * hm_size
        center_x_resized_int = int(center_x_resized_float)
        center_y_resized_int = int(center_y_resized_float)

        point = np.exp(-(np.arange(hm_size) - center_x_resized_int) ** 2 / (2 * sigma_x ** 2)).reshape(1, -1) * np.exp(
            -(np.arange(hm_size) - center_y_resized_int) ** 2 / (2 * sigma_y ** 2)).reshape(-1, 1)
        heatmap = np.maximum(heatmap, point)

        offset_x[center_y_resized_int, center_x_resized_int] = center_x_resized_float - center_x_resized_int
        offset_y[center_y_resized_int, center_x_resized_int] = center_y_resized_float - center_y_resized_int

        # creating patches
        y1 = max(center_y_resized_int - 5, 0)
        y2 = min(center_y_resized_int + 5, 128)
        x1 = max(center_x_resized_int - 5, 0)
        x2 = min(center_x_resized_int + 5, 128)
        object_size_x[y1:y2, x1:x2] = (x_max - x_min) * w_scale  # we store the size of the object in the resized image
        object_size_y[y1:y2, x1:x2] = (y_max - y_min) * h_scale

    offset = np.stack([offset_x, offset_y])
    object_size = np.stack([object_size_x, object_size_y])
    # heatmap = heatmap.reshape((1, 128, 128))
    return im_resized, heatmap, offset, object_size


if __name__ == '__main__':
    path_to_data = os.path.normpath("D:/")
    batch_size = 2
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VOCDataset(path_to_data, "train")
    print("Parsing...")
    dataset.process_dataset(save_name='processed_train.json')
    print("Ok")
    # l = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #
    # for im, hm, off, size in l:
    #     print(im)
    #     print(im.shape)
    #     print(off.shape)
    #     print(size.shape)
    #     break
