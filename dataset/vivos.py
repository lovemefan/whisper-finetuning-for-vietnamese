#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :Vivos.py
# @Time      :2023/1/27 15:30
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
import torch
import torchaudio
import whisper
from torchaudio import transforms


class VivosTraining(torch.utils.data.Dataset):
    def __init__(self, split="test", tokenizer=None, sample_rate=16000) -> None:
        super().__init__()

        root = f"data/vivos/{split}/waves"
        dataset = []
        with open(f"data/vivos/{split}/prompts.txt", "r") as f:
            a = f.read().split("\n")
        for i in a:
            x = i.find(" ")
            audio_id, text = i[:x], i[x + 1:]
            speaker = audio_id.split("_")[0]
            audio_path = f"{root}/{speaker}/{audio_id}.wav"
            # print(audio_path)
            if os.path.isfile(audio_path):
                dataset.append((audio_id, audio_path, text))

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
        audio_id, audio_path, text = self.dataset[id]
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
