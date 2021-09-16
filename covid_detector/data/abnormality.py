from os import truncate
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import PIL
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
import matplotlib.pyplot as plt

from covid_detector.util import *
from covid_detector.data.util import *                                   


DIR_DATA = Path('/kaggle/input')
DIR_DATA_TMP = Path('/kaggle/data/processed/abnormality')
JPG_IMG_SIZE = 512
CAT_NAMES = ['opacity', 'negative', 'typical', 'indeterminate', 'atypical']
MAX_NUM_INSTANCES = 100

BATCH_SIZE = 16
NUM_WORKERS = 4


def _scale_box(row, sz=256):
    if pd.isna(row.boxes):
        return np.nan

    scale0 = sz / row.dim0
    scale1 = sz / row.dim1
    boxes = eval(row.boxes)
    boxes_scaled = [scale_box(box, scale0, scale1) for box in boxes]
    return boxes_scaled


class AbnormalityDataset(Dataset):
    '''
    Object detection dataset for 5 abnormality classes:
    'opacity', 'negative', 'typical', 'indeterminate', and 'atypical'.
    '''

    def __init__(self, df, transform=None):
        self.df = df
        self._transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img = PIL.Image.open(r.pth_jpg).convert('RGB')
        img_width, img_height = img.size
        img = np.array(img)

        bboxes = []
        cls = []

        if isinstance(r.boxes, (list, np.ndarray)):
            for d_box in r.boxes:
                x, y = d_box['x'], d_box['y']
                width, height = d_box['width'], d_box['height']
                bbox = [y, x, y + height, x + width]
                bboxes.append(bbox)
                cls.append(CAT_NAMES.index('opacity'))

        for i, label_long in enumerate(STUDY_LEVEL_TARGETS_LONG):
            if r[label_long] == 1:
                bboxes.append([0, 0, img_height, img_width])
                short_label = STUDY_LEVEL_TARGETS_SHORT[i]
                cls.append(CAT_NAMES.index(short_label))

        if bboxes:
            bboxes = np.array(bboxes, ndmin=2, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        target = {'img_idx': idx,
                  'img_size': (img_width, img_height),
                  'bbox': bboxes,
                  'cls': cls}

        if self.transform:
            transformed = self.transform(
                image=img, bbox=target['bbox'], cls=target['cls'])
            img = transformed['image']
            target = {'img_idx': target['img_idx'],
                      'img_size': target['img_size'],
                      'bbox': transformed['bbox'],
                      'cls': transformed['cls']}

        return img, target

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t

    @staticmethod
    def show(img, target):
        fig, ax = plt.subplots(figsize=(8, 8))

        img_ = img.copy()

        for cls, bbox in zip(target['cls'], target['bbox']):
            y0, x0, y1, x1 = bbox.astype(np.int64)

            cv2.rectangle(img_, (x0, y0), (x1, y1), (255, 0, 0), 5)

            cv2.putText(img_, CAT_NAMES[cls], org=(x0 + 5, y0 + 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(255, 0, 0))

        ax.imshow(img_)

        return fig, ax


def abnormality_collate_fn(batch):
    batch_size = len(batch)
    image_size = 512

    img_b = torch.zeros(
        (batch_size, 3, image_size, image_size), dtype=torch.float32)
    bbox_b = torch.full(
        (batch_size, MAX_NUM_INSTANCES, 4), -1, dtype=torch.float32)
    cls_b = torch.full((batch_size, MAX_NUM_INSTANCES), -1, dtype=torch.int64)

    for i in range(batch_size):
        img, target = batch[i]
        bbox, cls = target['bbox'], target['cls']
        num_instances = len(bbox)
        num_elem = min(num_instances, MAX_NUM_INSTANCES)

        img_b[i] = img
        bbox_b[i, :num_elem] = torch.from_numpy(target['bbox'][:num_elem])
        cls_b[i, :num_elem] = torch.from_numpy(target['cls'][:num_elem])

    target_b = {'bbox': bbox_b, 'cls': cls_b,
                'img_size': None, 'img_scale': None}

    return img_b, target_b


class Abnormality(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__(args)
        args = vars(args) if args is not None else {}
        self.dir_data = args.get('dir_data', DIR_DATA)
        self.dir_data_tmp = args.get('dir_data_tmp', DIR_DATA_TMP)
        self.jpg_img_size = args.get('jpg_img_size', JPG_IMG_SIZE)
        self.batch_size = args.get('batch_size', BATCH_SIZE)
        self.num_workers = args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(args.get('gpu', None), (str, int))

        self.transform = A.Compose(
            [ToTensorV2(p=1)],
            bbox_params=A.BboxParams(format='coco', label_fields=['cls']))

        self.num_classes = len(CAT_NAMES)

    @staticmethod
    def add_to_argparser(parser):
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, 
                            help='Number of examples to operate on per forward step.')

        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                            help='Number of additional processes to load data.')
        return parser


        

    def prepare_data(self):
        if (self.dir_data_tmp / 'train.feather').exists():
            return

        dir_jpg = self.dir_data / \
            f'siim-covid19-resized-to-{self.jpg_img_size}px-jpg'
        assert dir_jpg.exists()

        dir_siim = self.dir_data / 'siim-covid19-detection'
        assert dir_siim.exists()

        df_train = prepare_train_examples(dir_siim, dir_jpg)
        df_train['boxes'] = df_train.apply(
            lambda row: _scale_box(row, sz=JPG_IMG_SIZE), axis=1)
        df_train['pth_jpg'] = df_train.pth_jpg.apply(lambda p: p.as_posix())
        self.dir_data_tmp.mkdir(parents=True, exist_ok=True)
        df_train.to_feather(self.dir_data_tmp / 'train.feather')

        # df_test  = prepare_test_examples(dir_siim, dir_jpg)

    def setup(self):
        df = pd.read_feather(self.dir_data_tmp / 'train.feather')
        val_cut = int(0.2 * len(df))

        self.val_dataset = AbnormalityDataset(
            df.iloc[:val_cut], transform=self.transform)
        self.train_dataset = AbnormalityDataset(
            df.iloc[val_cut:], transform=self.transform)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=abnormality_collate_fn,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.on_gpu)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=abnormality_collate_fn,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.on_gpu)


