#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :datasets.py
# @Time      :2023/1/27 15:28
# @Author    :lovemefan
# @email     :lovemefan@outlook.com

import json
import os
import numpy as np
import torch
import torchaudio
import whisper
import torchaudio.transforms as at


class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, texts = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            texts.append(f["text"])
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), "constant", constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        batch = {"labels": labels, "dec_input_ids": dec_input_ids}

        batch = {
            k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()
        }
        batch["input_ids"] = input_ids
        batch["texts"] = texts
        return batch
