import cv2


def preprocess(train, file_directory):

    # Ici on rempli un dictionnaire avec les labels pour chaque image

    dictionnary_labels_per_image = {}
    for image_id in train['image_id'].values:
        size = cv2.imread(file_directory+'{}.jpg'.format(image_id)).shape
        labels = train[train['image_id'] == image_id].labels.values[0].split(' ')
        letters = []
        bbox = []

        for i, l in enumerate(labels[::5]):
            letters.append(l)
            bbox.append([int(x) for x in labels[i * 5 + 1:i * 5 + 5]])

        dictionnary_labels_per_image[image_id] = {"letters": letters, "bbox": bbox, "size": size}

    return dictionnary_labels_per_image