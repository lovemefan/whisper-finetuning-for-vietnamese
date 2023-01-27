#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :finetuning.py
# @Time      :2023/1/27 15:37
# @Author    :lovemefan
# @email     :lovemefan@outlook.com

import os
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.__int__ import get_dataset
from dataset.datasets import WhisperDataCollatorWhithPadding


from dotenv import load_dotenv
import yaml

from model.model import WhisperModelModule


def load_config_file(path):
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    load_dotenv()
    config = load_config_file(os.environ.get("CONFIG_PATH", 'config/vi-base-vlsp2020.yaml'))
    print(config)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(config, config["train_dataset"], "train")
    valid_dataset = get_dataset(config, config["test_dataset"], "test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        collate_fn=WhisperDataCollatorWhithPadding(),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        collate_fn=WhisperDataCollatorWhithPadding(),
    )

    Path(config["log_output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["check_output_dir"]).mkdir(parents=True, exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=config["log_output_dir"],
        name=config["train_name"],
        version=config["train_id"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{config["check_output_dir"]}/checkpoint',
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1,  # all model save
    )

    checkpoint_callback_wer = ModelCheckpoint(dirpath=f'{config["check_output_dir"]}/logs',
                                              save_top_k=5, monitor="wer", mode="min",
                                              filename='{epoch}-{pesq:.4f}')

    checkpoint_callback_cer = ModelCheckpoint(dirpath=f'{config["check_output_dir"]}/logs',
                                              save_top_k=5, monitor="cer", mode="min",
                                              filename='{epoch}-{pesq:.4f}')

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch"), checkpoint_callback_wer,
                     checkpoint_callback_cer]
    model = WhisperModelModule(config, train_loader, valid_loader)

    trainer = Trainer(
        accelerator=DEVICE,
        max_epochs=config["num_train_epochs"],
        accumulate_grad_batches=config["gradient_accumulation_steps"],
        logger=tflogger,
        callbacks=callback_list,
        auto_scale_batch_size=True,
        auto_lr_find=True,
        resume_from_checkpoint=None if config.get("resume_from_checkpoint") == "" else config.get("resume_from_checkpoint")
    )

    trainer.fit(model)
