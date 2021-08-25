from os import truncate
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
import pytorch_lightning as pl

import PIL
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
import matplotlib.pyplot as plt

from covid_detector.util import *


DIR_DATA = Path('/kaggle/input')
DIR_DATA_TMP = Path('/kaggle/data/processed/abnormality')
JPG_IMG_SIZE = 256
CAT_NAMES = ['opacity', 'negative', 'typical', 'indeterminate', 'atypical']
MAX_NUM_INSTANCES = 100

BATCH_SIZE = 16
NUM_WORKERS = 4


class AbnormalityDataset(torch.utils.data.Dataset):
    '''
    Object detection dataset for 5 abnormality classes:
    'opacity', 'negative', 'typical', 'indeterminate', and 'atypical'.
    '''

    def __init__(self, df, transform=None):
        self.df = df
        self._transform = transform

    def __len__(self):
        return len(df)

    def __getitem__(self, idx):
        r = df.iloc[idx]

        img = PIL.Image.open(r.pth_jpg).convert('RGB')
        img_width, img_height = img.size
        img = np.array(img)

        bboxes = []
        cls = []

        if isinstance(r.boxes, list):
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
        fig, ax = plt.subplots(figsize=(10, 10))

        img_ = img.copy()

        for cls, bbox in zip(target['cls'], target['bbox']):
            y0, x0, y1, x1 = bbox.astype(np.int64)

            cv2.rectangle(img_, (x0, y0), (x1, y1), (255, 0, 0), 5)

            cv2.putText(img_, CAT_NAMES[cls], org=(x0 + 5, y0 + 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2,
                        color=(255, 0, 0))

        ax.imshow(img_)

        return fig, ax


def abnormality_collate_fn(batch):
    batch_size = len(batch)
    image_size = 512

    img_b = torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32)
    bbox_b = torch.full((batch_size, MAX_NUM_INSTANCES, 4), -1, dtype=torch.float32)
    cls_b = torch.full((batch_size, MAX_NUM_INSTANCES), -1, dtype=torch.int64)

    for i in range(batch_size):
        img, target = batch[i]
        bbox, cls = target['bbox'], target['cls']
        num_instances = len(bbox)
        num_elem = min(num_instances, MAX_NUM_INSTANCES)

        img_b[i] = img
        bbox_b[i, :num_elem] = torch.from_numpy(target['bbox'][:num_elem])
        cls_b[i, :num_elem] = torch.from_numpy(target['cls'][:num_elem])

    target_b = {'bbox': bbox_b, 'cls': cls_b}

    return img_b, target_b


class Abnormality(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__(args)
        args = vars(args) if args is not None else {}
        self.dir_data = args.get('dir_data', DIR_DATA)
        self.dir_data_tmp = args.get('dir_data_tmp', DIR_DATA_TMP)
        self.jpg_img_size = args.get('jpg_img_size', JPG_IMG_SIZE)
        
    def prepare_data(self):
        dir_jpg = self.dir_data / f'siim-covid19-resized-to-{self.jpg_img_size}px-jpg'
        assert dir_jpg.exists()
        
        dir_siim = self.dir_data / 'siim-covid19-detection'
        assert dir_siim.exists()
            
        df_train = _prepare_train_examples(dir_siim, dir_jpg)
        # df_test  = _prepare_test_examples(dir_siim, dir_jpg)

        self.dir_data_tmp.mkdir(parents=True, exist_ok=True)
        df_train.to_feather(self.dir_data_tmp / 'train.feather')

    def setup(self):
        df = pd.read_csv(self.dirname / 'train.feather')
        dataset = AbnormalityDataset(df, transform=self.transform)
        num_examples = len(dataset)
        num_train_examples = int(0.8 * num_examples)
        num_valid_examples = num_examples - num_train_examples

        self.train_dataset, self.val_dataset = random_split(
            dataset, lengths=[num_train_examples, num_valid_examples])



