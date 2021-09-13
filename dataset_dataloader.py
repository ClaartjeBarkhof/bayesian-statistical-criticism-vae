from datasets import load_from_disk
# from transformers import AutoTokenizer, AutoModel
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np


# Parts of this code are taken from https://github.com/probabll/mixed-rv-vae/blob/master/data.py

# TODO: write this piece of code
# class LanguageDatasetDataLoader:
#     def __init__(self, language_dataset_name, tokeniser_name):
#         # tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
#         # model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
#
#     def train_loader(self):
#         pass
#
#     def valid_loader(self):
#         pass
#
#     def test_loader(self):
#         pass

def greater_than_zero(x):
    return x > 0


def to_float(x):
    return x.float()


class ImageDataset:
    def __init__(self, args):

        self.image_dataset_name = args.image_dataset_name
        self.image_w = args.image_w
        self.image_h = args.image_h
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # -------------------------------
        # TRANSFORMS
        # -------------------------------

        self.img_transforms = []

        # RESIZE
        if not self.image_h == self.image_w == 28:
            self.img_transforms += [transforms.Resize((self.image_h, self.image_w))]

        # TO TENSOR + NORMALISE
        self.img_transforms += [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]

        # BINARISE
        if self.image_dataset_name == "bmnist":
            self.img_transforms += [greater_than_zero, to_float]

        # COMPOSE
        self.img_transforms = transforms.Compose(self.img_transforms)

        # -------------------------------
        # LOADING DATA SETS
        # -------------------------------

        # MNIST + BMNIST
        # Sizes: train, valid, test = 55000, 5000, 10000
        if self.image_dataset_name in ["mnist", "bmnist"]:
            self.train_set = Subset(
                datasets.MNIST(self.data_dir, train=True, download=True, transform=self.img_transforms),
                indices=range(55000))
            self.valid_set = Subset(
                datasets.MNIST(self.data_dir, train=True, download=True, transform=self.img_transforms),
                indices=range(55000, 60000))
            self.test_set = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.img_transforms)

    def train_loader(self):
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader

    def valid_loader(self):
        valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        return valid_loader

    def test_loader(self):
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return test_loader

# if __name__ == "__main__":
