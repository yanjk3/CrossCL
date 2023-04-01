import numpy as np
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.datasets import DatasetFolder
import os.path as osp
import os
from PIL import Image


class ImageFolder(DatasetFolder):
    def __init__(self, root, img_size=224., transform=None, target_transform=None,
                 crop_transform=None, flip_transform=None, ending_transform=None,
                 loader=default_loader, is_valid_file=None, dataset='imagenet'):
        self.dataset = dataset
        if dataset == 'imagenet':
            super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                              transform=transform,
                                              target_transform=target_transform,
                                              is_valid_file=is_valid_file)
        elif dataset == 'coco':
            self.samples = [osp.join(root, i) for i in os.listdir(root)]

        self.img_size = img_size
        self.base_transform = transform
        self.crop_transform = crop_transform
        self.flip_transform = flip_transform
        self.ending_transform = ending_transform
        # window size48 stride32 by default, adjust accordingly with img_size
        self.q_anchors = self.sliding_windows(48 * self.img_size / 224., 32 * self.img_size / 224.)

    def sliding_windows(self, size, stride):
        q_anchors = list()
        for i in range(6):
            for j in range(6):
                y0, x0 = i * stride, j * stride
                y1, x1 = y0 + size, x0 + size
                if x1 >= self.img_size or y1 >= self.img_size:
                    continue
                q_anchors.append(np.array([x0, y0, x1, y1]))
        q_anchors = np.array(q_anchors, dtype=np.float)
        return q_anchors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.dataset == 'imagenet':
            path, target = self.samples[index]
            sample = self.loader(path)
        elif self.dataset == 'coco':
            name = self.samples[index]
            sample = Image.open(name).convert("RGB")
        else:
            raise NotImplementedError

        if self.base_transform is not None:
            q, k = self.base_transform(sample), self.base_transform(sample)
        else:
            q, k = sample, sample

        if self.flip_transform is not None:
            q, is_flip = self.flip_transform(q)
            k, _ = self.flip_transform(k, is_flip=is_flip)

        q, (t_q, l_q, h_q, w_q) = self.crop_transform(q)
        k, (t_k, l_k, h_k, w_k) = self.crop_transform(k)
        b_q, r_q = t_q + h_q, l_q + w_q
        b_k, r_k = t_k + h_k, l_k + w_k

        if self.ending_transform is not None:
            q, k = self.ending_transform(q), self.ending_transform(k)

        k_anchors, anchors_label = self.coordinate_trans((l_q, t_q, r_q, b_q), (l_k, t_k, r_k, b_k))

        return [q, k], [self.q_anchors.copy(), k_anchors, anchors_label]

    def coordinate_trans(self, crop_area_q, crop_area_k):
        k_anchors = list()
        anchors_label = list()
        for anchor in self.q_anchors:
            x0, y0, x1, y1 = self.anchor_mapping(anchor, crop_area_q, crop_area_k)
            if x0 == -1:
                anchor_label = -1
                x0 = y0 = 0  # fake anchor
                x1 = y1 = 1
            else:
                anchor_label = 0
            anchors_label.append(anchor_label)
            k_anchors.append(np.array([x0, y0, x1, y1]))
        k_anchors = np.array(k_anchors, dtype=np.float)
        anchors_label = np.array(anchors_label, dtype=np.float)
        return k_anchors, anchors_label

    def anchor_mapping(self, anchor, loc_q, loc_k):
        """
        Args:
            anchor: (x0, y0, x1, y1) The coordinates of anchors in image q
            loc_q: (x0, y0, x1, y1) The cropped region of image q w.r.t. the original image
            loc_k: (x0, y0, x1, y1) The cropped region of image k w.r.t. the original image
        Returns:
            tuple: (x0, y0, x1, y1) if transformed region falls into the boundary of image k, or (-1, -1, -1, -1) if not
        """
        img_size = self.img_size
        x0, y0, x1, y1 = anchor
        l_q, t_q, r_q, b_q = loc_q
        l_k, t_k, r_k, b_k = loc_k
        # Step 1: transfer anchor's coordinates (relative to q) to absolute
        l_a = (r_q - l_q) * x0 / img_size + l_q
        r_a = (r_q - l_q) * x1 / img_size + l_q
        t_a = (b_q - t_q) * y0 / img_size + t_q
        b_a = (b_q - t_q) * y1 / img_size + t_q
        # Step 2: transfer anchor's absolute coordinates to relative to k
        x0 = (l_a - l_k) * img_size / (r_k - l_k)
        x1 = (r_a - l_k) * img_size / (r_k - l_k)
        y0 = (t_a - t_k) * img_size / (b_k - t_k)
        y1 = (b_a - t_k) * img_size / (b_k - t_k)
        # Step 3: check whether it's out of boundary
        if x0 < 0 or y0 < 0 or x1 >= img_size or y1 >= img_size:
            x0 = y0 = x1 = y1 = -1
        return x0, y0, x1, y1
