
from pathlib import Path
import pytorch_lightning as pl

from pathlib import Path
import argparse
import pandas as pd
import pytorch_lightning as pl

from covid_detector.util import *


DIR_DATA = Path('/kaggle/input')
STUDY_LEVEL_TARGETS_LONG = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 
                            'Atypical Appearance']
STUDY_LEVEL_TARGETS_SHORT = ["negative", "typical", "indeterminate", "atypical"]
JPG_IMG_SIZE = 256



def _prepare_train_examples(dir_siim, dir_jpg):
    df_jpg = pd.read_csv(dir_jpg / 'meta.csv')
    df_jpg['pth_jpg'] = df_jpg.apply(lambda r: dir_jpg / r.split / f'{r.image_id}.jpg', axis=1)

    df_study = pd.read_csv(dir_siim / 'train_study_level.csv')
    df_study['id'] = df_study.id.apply(lambda id: id.replace('_study', ''))
    df_study.rename({'id': 'study_id'}, axis=1, inplace=True)

    df_image = pd.read_csv(dir_siim / 'train_image_level.csv')
    df_image['id'] = df_image.id.apply(lambda id: id.replace('_image', ''))
    df_image.drop('label', axis=1, inplace=True)
    df_image.rename({'id': 'image_id', 'StudyInstanceUID': 'study_id'}, axis=1, inplace=True)

    df = pd.merge(df_image, df_study, left_on='study_id', right_on='study_id', how='inner')
    df = df[['image_id', 'study_id', 'boxes'] + STUDY_LEVEL_TARGETS_LONG]

    df = pd.merge(df, df_jpg, left_on='image_id', right_on='image_id', how='inner')
    return df


def _prepare_test_examples(dir_siim, dir_jpg):
    df_jpg = pd.read_csv(dir_jpg / 'meta.csv')
    df_jpg['pth_jpg'] = df_jpg.apply(lambda r: dir_jpg / r.split / f'{r.image_id}.jpg', axis=1)

    df_test = pd.read_csv(dir_siim / 'sample_submission.csv')
    is_study = df_test.id.apply(lambda id: id.endswith('_study'))
    study_ids = df_test[is_study].id.apply(lambda id: id.replace('_study', '')).values

    study_id_list = []
    img_id_list = []
    for study_id in study_ids:
        for dicom_pth in (dir_siim / 'test' / study_id).rglob('*.dcm'):
            img_id = dicom_pth.stem
            study_id_list.append(study_id)
            img_id_list.append(img_id)

    df = pd.DataFrame({'image_id': img_id_list, 'study_id': study_id_list})
    df = pd.merge(df, df_jpg, left_on='image_id', right_on='image_id', how='inner')
    return df


class Abnormality(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__(args)
        args = vars(args) if args is not None else {}
        self.dir_data = args.get('dir_data', DIR_DATA)
        self.jpg_img_size = args.get('jpg_img_size', JPG_IMG_SIZE)
        
    def prepare_data(self):
        dir_jpg = self.dir_data / f'siim-covid19-resized-to-{self.jpg_img_size}px-jpg'
        assert dir_jpg.exists()
        
        dir_siim = self.dir_data / 'siim-covid19-detection'
        assert dir_siim.exists()
            
        _prepare_train_examples(dir_siim, dir_jpg)
        _prepare_test_examples(dir_siim, dir_jpg)
