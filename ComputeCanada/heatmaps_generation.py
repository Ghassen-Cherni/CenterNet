import numpy as np

def gaussian_heatmap(sigma, center_x, center_y, heatmap):
    #     for i in range(heatmap.shape[0]):
    #         for j in range(heatmap.shape[1]):
    #             gaussian = np.exp(-((j-center_x)*(j-center_x)+(i-center_y)*(i-center_y))/(2*sigma*sigma))
    #             if gaussian > heatmap[i, j]:
    #                 heatmap[i,j]=gaussian

    point = np.exp(-(np.arange(128) - center_x) ** 2 / (2 * sigma * sigma)).reshape(1, -1
                                                                                    ) * np.exp(
        -(np.arange(128) - center_y) ** 2 / (2 * sigma * sigma)).reshape(-1, 1)
    heatmap = np.maximum(heatmap, point)
    return heatmap


def generate_heatmap_offset(image_id, sigma, dictionnary_labels_per_image):
    heatmap = np.zeros((128, 128))
    offset_x = np.zeros((128, 128))
    offset_y = np.zeros((128, 128))
    object_size_x = np.zeros((128, 128))
    object_size_y = np.zeros((128, 128))
    bbox = dictionnary_labels_per_image[image_id]["bbox"]
    img_shape = dictionnary_labels_per_image[image_id]["size"]

    for i in range(len(bbox)):
        center_x = bbox[i][0] + bbox[i][2] // 2
        center_y = bbox[i][1] + bbox[i][3] // 2

        center_x_resized_float = center_x / img_shape[1] * 128
        center_y_resized_float = center_y / img_shape[0] * 128

        center_x_resized_int = int(center_x_resized_float)
        center_y_resized_int = int(center_y_resized_float)

        heatmap = gaussian_heatmap(sigma, center_x_resized_int, center_y_resized_int, heatmap)
        offset_x[center_y_resized_int, center_x_resized_int] = center_x_resized_float - center_x_resized_int
        offset_y[center_y_resized_int, center_x_resized_int] = center_y_resized_float - center_y_resized_int
        object_size_x[center_y_resized_int, center_x_resized_int] = (bbox[i][2] // 2)
        object_size_y[center_y_resized_int, center_x_resized_int] = (bbox[i][3] // 2)

    return heatmap, offset_x, offset_y, object_size_x, object_size_y