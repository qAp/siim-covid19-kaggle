
import PIL
import numpy as np
import pandas as pd
import torch

import matplotlib
import matplotlib.pyplot as plt



STUDY_LEVEL_TARGETS_LONG = ['Negative for Pneumonia', 
                            'Typical Appearance', 
                            'Indeterminate Appearance',
                            'Atypical Appearance']

STUDY_LEVEL_TARGETS_SHORT = ["negative",
                             "typical", 
                             "indeterminate", 
                             "atypical"]


def prepare_train_examples(dir_siim, dir_jpg):
    df_jpg = pd.read_csv(dir_jpg / 'meta.csv')
    df_jpg['pth_jpg'] = df_jpg.apply(
        lambda r: dir_jpg / r.split / f'{r.image_id}.jpg', axis=1)

    df_study = pd.read_csv(dir_siim / 'train_study_level.csv')
    df_study['id'] = df_study.id.apply(lambda id: id.replace('_study', ''))
    df_study.rename({'id': 'study_id'}, axis=1, inplace=True)

    df_image = pd.read_csv(dir_siim / 'train_image_level.csv')
    df_image['id'] = df_image.id.apply(lambda id: id.replace('_image', ''))
    df_image.drop('label', axis=1, inplace=True)
    df_image.rename(
        {'id': 'image_id', 'StudyInstanceUID': 'study_id'}, axis=1, inplace=True)

    df = pd.merge(df_image, df_study, left_on='study_id',
                  right_on='study_id', how='inner')
    df = df[['image_id', 'study_id', 'boxes'] + STUDY_LEVEL_TARGETS_LONG]

    df = pd.merge(df, df_jpg, left_on='image_id',
                  right_on='image_id', how='inner')
    return df


def prepare_test_examples(dir_siim, dir_jpg):
    df_jpg = pd.read_csv(dir_jpg / 'meta.csv')
    df_jpg['pth_jpg'] = df_jpg.apply(
        lambda r: dir_jpg / r.split / f'{r.image_id}.jpg', axis=1)

    df_test = pd.read_csv(dir_siim / 'sample_submission.csv')
    is_study = df_test.id.apply(lambda id: id.endswith('_study'))
    study_ids = df_test[is_study].id.apply(
        lambda id: id.replace('_study', '')).values

    study_id_list = []
    img_id_list = []
    for study_id in study_ids:
        for dicom_pth in (dir_siim / 'test' / study_id).rglob('*.dcm'):
            img_id = dicom_pth.stem
            study_id_list.append(study_id)
            img_id_list.append(img_id)

    df = pd.DataFrame({'image_id': img_id_list, 'study_id': study_id_list})
    df = pd.merge(df, df_jpg, left_on='image_id',
                  right_on='image_id', how='inner')
    return df


def scale_box(box, scale0=1, scale1=1):
    box = {k: scale1 * v if k in ('x', 'width')
           else scale0 * v for k, v in box.items()}
    box = {k: np.around(v, decimals=5) for k, v in box.items()}
    return box


def _scale_box(row, sz=256):
    if pd.isna(row.boxes):
        return np.nan

    scale0 = sz / row.dim0
    scale1 = sz / row.dim1
    boxes = eval(row.boxes)
    boxes_scaled = [scale_box(box, scale0, scale1) for box in boxes]
    return boxes_scaled
