import csv
import pickle
import threading
from typing import List
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from submodules.FileTools.WordOperator import str_format
from src.data_process import get_dataloader
from src.train_process import DL_Model


class TaskThread(threading.Thread):
    def __init__(
        self,
        model: DL_Model,
        mode: str = 'train',
        loader: DataLoader or None = None,
        val_loader: DataLoader or None = None,
    ):
        threading.Thread.__init__(self)
        self.model = model
        self.mode = mode
        self.loader = loader
        self.val_loader = val_loader
        self.result = None

    def run(self):
        if self.mode == 'train':
            self.result = self.model.training(self.loader, self.val_loader)
        elif self.mode == 'test':
            self.result = self.model.testing(self.loader)

    def execute(self):
        self.start()
        try:
            self.join()
        except KeyboardInterrupt:
            self.model.earlyStop = self.model.epoch + 1


def train(model: DL_Model = None, loader: List[DataLoader] = None, **kwargs):
    if model is None:
        model = DL_Model()

        if loader is not None:
            train_loader, valid_loader = loader
        else:
            train_loader, valid_loader, speaker_num = get_dataloader(
                data_dir=kwargs['data_dir'], batch_size=model.BATCH_SIZE, n_workers=kwargs['n_workers']
            )
            model.net_config(net_parameter=speaker_num)

    task = TaskThread(model=model, loader=train_loader, val_loader=valid_loader)
    task.execute()


def pre_train(model_pickle: str, loader: List[DataLoader] = None, **kwargs):
    model = DL_Model()
    model.load_model(model_pickle)

    if loader is not None:
        train_loader, valid_loader = loader
    else:
        train_loader, valid_loader, speaker_num = get_dataloader(
            data_dir=kwargs['data_dir'], batch_size=model.BATCH_SIZE, n_workers=kwargs['n_workers']
        )

    model.load_model(model_pickle)
    task = TaskThread(model=model, loader=train_loader, val_loader=valid_loader)
    task.execute()


def full_train(model: DL_Model = None, loader: DataLoader = None, **kwargs):
    if model is None:
        model = DL_Model()

    if loader is None:
        loader, speaker_num = get_dataloader(
            data_dir=kwargs['data_dir'], mode='full', batch_size=model.BATCH_SIZE, n_workers=kwargs['n_workers']
        )
        model.net_config(net_parameter=speaker_num)

    model.saveDir = './out/full/'
    task = TaskThread(model=model, loader=loader)
    task.execute()


def full_pre_train(model_pickle: str, model: DL_Model = None, loader: DataLoader = None, **kwargs):
    if model is None:
        model = DL_Model()

    model.load_model(model_pickle)

    if loader is None:
        loader, speaker_num = get_dataloader(
            data_dir=kwargs['data_dir'], mode='full', batch_size=model.BATCH_SIZE, n_workers=kwargs['n_workers']
        )

    task = TaskThread(model=model, loader=loader)
    task.execute()


if __name__ == '__main__':
    loader_params = {
        'data_dir': './Data/Dataset',
        'n_workers': 8,
    }

    train(**loader_params)

    # pre_train()
    # full_train()
    # full_pre_train()
