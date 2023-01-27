#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :__int__.py
# @Time      :2023/1/27 16:19
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
from dataset.vivos import VivosTraining
from dataset.vlsp2019 import VLSP2019Training
from dataset.vlsp2020 import VLSP2020Training


def get_dataset(config, dataset_name: str, split: str, *args, **kwargs):
    if dataset_name == 'vlsp2019':
        dataset = VLSP2019Training(split, *args, **kwargs)
    elif dataset_name == 'vlsp2020':
        dataset = VLSP2020Training(split, manifest_path=config['manifest_path'], *args, **kwargs)
    elif dataset_name == 'vivos':
        dataset = VivosTraining(split, *args, **kwargs)
    else:
        raise ValueError(f"{dataset_name} not support")

    return dataset
