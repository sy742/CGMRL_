import torch
import numpy as np
import h5py

class creat_adj(object):
    def __init__(self, img_features_h5_path):
        super().__init__()
        self.img_features_h5_path = img_features_h5_path

    def area(self, boxes):
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return area

    def boxlist_iou(self, boxlist1, boxlist2):
        """Compute the intersection over union of two set of boxes.
        The box order must be (xmin, ymin, xmax, ymax).
        Arguments:
        box1: (BoxList) bounding boxes, sized [N,4].
        box2: (BoxList) bounding boxes, sized [M,4].
        Returns:
        (tensor) iou, sized [N,M].
        """
        # N = boxlist1.shape[0]
        # M = boxlist2.shape[1]
        area1 = self.area(boxlist1)

        area2 = self.area(boxlist2)
        lt = np.max(boxlist1[:, None, :2], boxlist2[:, :2])  # [N,M,2]
        rb = np.min(boxlist1[:, None, 2:], boxlist2[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        iou = inter / (area1[:, None] + area2 - inter)
        return iou
    def get_feature(self):
        with h5py.File(self.text_features_h5_path, "r") as text_features_hdf:
            for f_key in self.feature_keys:
                features[f_key] = text_features_hdf[f_key][text_feature_index]
            image_id = text_features_hdf["img_ids"][text_feature_index]
        img_feature_index = self.img_feature_id_l.index(image_id)
        with h5py.File(self.img_features_h5_path, "r") as features_hdf:
            self.pos_boxes = np.array(features_hdf.get("pos_boxes"))
        with h5py.File(self.img_features_h5_path, "r") as features_hdf:
            image_features = features_hdf["image_features"][
                             self.pos_boxes[img_feature_index][0]: self.pos_boxes[img_feature_index][1], :]
            bounding_boxes = [features_hdf["image_bb"][feature_idx]
                              for feature_idx in range(self.pos_boxes[index][0], self.pos_boxes[index][1])]
