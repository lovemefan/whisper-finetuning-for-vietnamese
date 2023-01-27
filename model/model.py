#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :whiper.py
# @Time      :2023/1/27 15:24
# @Author    :lovemefan
# @email     :lovemefan@outlook.com

import torch
from torch import nn
import whisper

from pytorch_lightning import LightningModule

import evaluate

from transformers import AdamW, get_linear_schedule_with_warmup


class WhisperModelModule(LightningModule):
    def __init__(
        self,
        config,
        train_dataloader=None,
        eval_dataloader=None,
    ) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(
            language=config["lang"], without_timestamps=True
        )
        self.model = whisper.load_model(config["model_name"])
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=config["lang"], task=self.options.task
        )

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.config = config
        self.trainloader = train_dataloader
        self.evaloader = eval_dataloader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("wer", wer, on_step=True, prog_bar=True, logger=True)

        return {"cer": cer, "wer": wer, "loss": loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            eps=self.config["adam_epsilon"],
        )
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.t_total,
        )
        self.scheduler = scheduler

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.t_total = (
                (len(self.trainloader.dataset) // (self.config["batch_size"]))
                // self.config["gradient_accumulation_steps"]
                * float(self.config["num_train_epochs"])
            )

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.evaloader
