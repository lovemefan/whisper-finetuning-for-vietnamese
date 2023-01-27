#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :vlsp2020.py
# @Time      :2023/1/27 16:04
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
import os.path

import torch
import torchaudio
import whisper
import json
from torchaudio import transforms


class VLSP2020Training(torch.utils.data.Dataset):
    def __init__(self, split="test", manifest_path="", tokenizer=None, sample_rate=16000) -> None:
        super().__init__()

        root_path = os.path.join(manifest_path, split, "manifest.tsv")
        dataset = []
        with open(root_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            audio_path, text = line.split('\t')
            if os.path.exists(audio_path):
                dataset.append((audio_path, text))

        self.dataset = dataset
        # self.dataset = [self.dataset[i] for i in range(100)]
        self.sample_rate = sample_rate
        self.options = whisper.DecodingOptions(language="vi", without_timestamps=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="vi", task=self.options.task
        )

    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = transforms.Resample(sr, sample_rate)(waveform)
        return waveform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        audio_path, text = self.dataset[id]

        audio = self.load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text_token = [
                         *self.tokenizer.sot_sequence_including_notimestamps
                     ] + self.tokenizer.encode(text)
        labels = text_token[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text_token,
            "text": text,
        }
