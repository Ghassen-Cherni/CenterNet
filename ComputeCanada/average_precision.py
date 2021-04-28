from mean_average_precision import MetricBuilder
from skimage.feature import peak_local_max
import numpy as np

def get_average_precision(hm_gt, reg_gt, hm_pred, reg_pred, min_distance, iou_threshold):
    """
    Input:
        hm_gt: batched ground truth heatmaps. Size = (batch_size, s, s), typically s=128
        hm_pred: batched predicted heatmaps. Size = (batch_size, s, s), typically s=128
        reg_gt: batched ground truth regression. Size = (batch_size, 4, s,s), typically s=128. Second dimension is: offset x, offset y, size x, size y
        reg_pred: batched predicted regression. Size = (batch_size, 4, s,s), typically s=128. Second dimension is: offset x, offset y, size x, size y
        min_distance: int, for peak local max estimation
        iou_threshold: float in [0,1]

    """
    hm_gt = hm_gt.cpu()
    reg_gt = reg_gt.cpu()
    # hm_pred = hm_pred.cpu().detach()
    # reg_pred = reg_pred.cpu().detach()
    hm_pred = hm_pred.cpu()
    reg_pred = reg_pred.cpu()
    batch_size = int(hm_gt.shape[0])
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for i in range(batch_size):
        gt_bboxes = get_resized_bboxes(hm_gt[i], reg_gt[i], min_distance=min_distance, status="gt")
        pred_bboxes = get_resized_bboxes(hm_pred[i], reg_pred[i], min_distance=min_distance, status="pred")
        metric_fn.add(pred_bboxes, gt_bboxes)

    return metric_fn.value(iou_thresholds=iou_threshold)['mAP']


def get_resized_bboxes(hm, reg, min_distance, status):
    """
    THis function takes as input heatmaps and outputs the detected bounding boxes in the (512,512) images

    Input:
        hm: heatmap of centers, of size (s,s), typically s=128
        reg: regression heatmaps, size (4,s,s), typically s=128. In this order: offset_x, offset_y, size_x, size_y
        min_distance: int, min distance for the skimage peak local max functions

    Output:
        out: list of list. Each list contains a detected bbox, format [x1, y1, x2, y2]
    """
    h0, w0 = (512, 512)
    if len(hm.shape) > 2:
        hm = hm.squeeze()
    hm_h, hm_w = hm.shape

    ratio_x = w0 / hm_w
    ratio_y = h0 / hm_h
    out = []

    centers = peak_local_max(hm.numpy(), min_distance=min_distance)
    for center in centers:
        y, x = center
        off_x, off_y, size_x, size_y = reg[:, y, x]
        off_x *= ratio_x
        off_y *= ratio_y
        y = ratio_y * y + off_y
        x = ratio_x * x + off_x
        if status == "gt":
            out.append([x.item(),y.item(), (x + size_x).item(), (y+size_y).item(), 0, 0, 0])
        elif status == "pred":
            out.append([x.item(),y.item(), (x + size_x).item(), (y+size_y).item(), 0, 1])
    return np.array(out).astype(int)

###################
# Un exemple
###################

# model = CenterNet()
# im, hm, off, siz = next(iter(train_dataloader))

# with torch.no_grad():
#     hm_pred, size_pred, off_pred = model(im.float())
#
# #concatenate offset and size heatmaps
# reg = torch.cat((off, siz), 1)
# reg_pred = torch.cat([off_pred, size_pred], 1)
#
# get_average_precision(hm, reg, hm_pred, reg_pred, 5, 0.4)

