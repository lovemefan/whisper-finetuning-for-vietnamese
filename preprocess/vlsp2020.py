#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :vlsp2020.py
# @Time      :2023/1/27 16:38
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
import os
from argparse import ArgumentParser


def generate_manifest(root_path, output_path):
    manifest = []
    for dir_list in os.listdir(root_path):
        for file in os.listdir(os.path.join(root_path, dir_list)):
            if file.endswith('.wav'):
                file_path = os.path.join(root_path, dir_list, file.replace('wav', 'txt'))
                with open(file_path, 'r', encoding='utf-8') as txt:
                    text = txt.read().strip()
                manifest.append(f"{os.path.join(root_path, dir_list, file)}\t{text}")

    train_set = manifest[:-1000]
    test_set = manifest[-1000:]

    if not os.path.exists(os.path.join(output_path, 'train')):
        os.makedirs(os.path.join(output_path, 'train'))

    if not os.path.exists(os.path.join(output_path, 'test')):
        os.makedirs(os.path.join(output_path, 'test'))

    with open(os.path.join(output_path, 'train', 'manifest.tsv'), 'w', encoding='utf-8') as file:
        file.write("\n".join(train_set))

    with open(os.path.join(output_path, 'test', 'manifest.tsv'), 'w', encoding='utf-8') as file:
        file.write("\n".join(test_set))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    generate_manifest(args.dataset_path, args.output_path)
