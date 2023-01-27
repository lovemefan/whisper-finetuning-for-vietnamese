#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :vlsp2019.py
# @Time      :2023/1/27 15:31
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
import torch
import torchaudio
import whisper
import json
from torchaudio import transforms


class VLSP2019Training(torch.utils.data.Dataset):
    def __init__(self, split="test", tokenizer=None, sample_rate=16000) -> None:
        super().__init__()

        root = f"data/vivos/{split}/waves"
        dataset = []
        with open(f"data/InfoReTechnology-vi/415hours/{split}/data_book_train_relocated.json", "r") as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            audio_path, text = line['key'], line['text']
            audio_path = audio_path.replace('/home/asr/data/phase2/', 'data/InfoReTechnology-vi/415hours/')

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
