# Claartje: parts of this code are taken from Wilker's code:
# https://github.com/probabll/mixed-rv-vae/blob/master/data.py

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import torch.utils.data as data_utils

import os

# from datasets import load_from_disk
# from transformers import AutoTokenizer, AutoModel
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


class ImageDataset:
    def __init__(self, args):

        self.image_dataset_name = args.image_dataset_name
        self.image_w_h = args.image_w_h
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # -------------------------------
        # TRANSFORMS
        # -------------------------------

        self.img_transforms = []

        image_dataset_options = ["bmnist", "mnist", "fmnist"]
        assert self.image_dataset_name.lower() in image_dataset_options, \
            f"Image dataset name not one of the valid options {image_dataset_options}"

        # RESIZE
        if not self.image_w_h == 28:
            self.img_transforms += [transforms.Resize((self.image_w_h, self.image_w_h))]

        # TO TENSOR
        # + NORMALISE? transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        # TODO: not sure when to use this, perhaps if doing some form of mean regression?
        self.img_transforms += [transforms.ToTensor()]

        # BINARISE
        if self.image_dataset_name == "bmnist":
            self.img_transforms += [greater_than_zero, to_float]

        # COMPOSE
        self.img_transforms = transforms.Compose(self.img_transforms)

        # -------------------------------
        # LOADING DATA SETS
        # -------------------------------

        # MNIST
        # Sizes: train, valid, test = 55000, 5000, 10000
        if self.image_dataset_name == "mnist":
            self.train_set = Subset(
                datasets.MNIST(self.data_dir, train=True, download=True, transform=self.img_transforms),
                indices=range(55000))
            self.valid_set = Subset(
                datasets.MNIST(self.data_dir, train=True, download=True, transform=self.img_transforms),
                indices=range(55000, 60000))
            self.test_set = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.img_transforms)

        # FMNIST
        # Original size: train, test = 60000, 10000
        # Manipulated sizes: train, validation, test = = 55000, 5000, 10000
        elif self.image_dataset_name == "fmnist":
            self.train_set = Subset(
                datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=self.img_transforms),
                indices=range(55000))
            self.valid_set = Subset(
                datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=self.img_transforms),
                indices=range(55000, 60000))
            self.test_set = datasets.FashionMNIST(self.data_dir, train=False, download=True,
                                                  transform=self.img_transforms)
        # BMNIST
        else:
            with open(f'{self.data_dir}/BMNIST/binarized_mnist_train.amat') as f:
                lines = f.readlines()
            x_train = lines_to_np_array(lines).astype('float32')
            with open(f'{self.data_dir}/BMNIST/binarized_mnist_valid.amat') as f:
                lines = f.readlines()
            x_val = lines_to_np_array(lines).astype('float32')
            with open(f'{self.data_dir}/BMNIST/binarized_mnist_test.amat') as f:
                lines = f.readlines()

            x_test = lines_to_np_array(lines).astype('float32')

            # KNN predicted ys
            y_train = torch.load(f'{self.data_dir}/BMNIST/binarized_mnist_train_KNN_pred_y.pt')
            y_val = torch.load(f'{self.data_dir}/BMNIST/binarized_mnist_valid_KNN_pred_y.pt')
            y_test = torch.load(f'{self.data_dir}/BMNIST/binarized_mnist_test_KNN_pred_y.pt')

            # pytorch data loader
            self.train_set = data_utils.TensorDataset(torch.from_numpy(x_train).reshape(
                len(x_train), 1, 28, 28).float(), y_train)
            self.valid_set = data_utils.TensorDataset(torch.from_numpy(x_val).reshape(
                len(x_val), 1, 28, 28).float(), y_val)
            self.test_set = data_utils.TensorDataset(torch.from_numpy(x_test).reshape(
                len(x_test), 1, 28, 28).float(), y_test)

            # idle y's
            # y_train = np.zeros((x_train.shape[0], 1))
            # y_val = np.zeros((x_val.shape[0], 1))
            # y_test = np.zeros((x_test.shape[0], 1))
            #
            # # pytorch data loader
            # self.train_set = data_utils.TensorDataset(
            #     torch.from_numpy(x_train).reshape(len(x_train), 1, 28, 28).float(), torch.from_numpy(y_train))
            # self.valid_set = data_utils.TensorDataset(torch.from_numpy(x_val).reshape(len(x_val), 1, 28, 28).float(),
            #                                           torch.from_numpy(y_val))
            # self.test_set = data_utils.TensorDataset(torch.from_numpy(x_test).reshape(len(x_test), 1, 28, 28).float(),
            #                                          torch.from_numpy(y_test))

    def train_loader(self, shuffle=True, batch_size=None, num_workers=None):
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size if batch_size is None else batch_size,
                                  shuffle=shuffle,
                                  num_workers=self.num_workers if num_workers is None else num_workers)
        return train_loader

    def valid_loader(self, num_workers=None, batch_size=None, shuffle=False):
        valid_loader = DataLoader(self.valid_set, shuffle=shuffle,
                                  batch_size=self.batch_size if batch_size is None else batch_size,
                                  num_workers=self.num_workers if num_workers is None else num_workers)
        return valid_loader

    def test_loader(self, batch_size=None, shuffle=False, num_workers=None):
        test_loader = DataLoader(self.test_set,
                                 batch_size=self.batch_size if batch_size is None else batch_size,
                                 shuffle=shuffle,
                                 num_workers=self.num_workers if num_workers is None else num_workers)
        return test_loader

    def get_train_validation_loaders(self):
        loaders = dict(train=self.train_loader(), valid=self.valid_loader())
        return loaders


def greater_than_zero(x):
    return x > 0


def to_float(x):
    return x.float()


def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])

if __name__=="__main__":
    from arguments import prepare_parser
    import matplotlib.pyplot as plt

    config = prepare_parser(jupyter=False, print_settings=True)
    dataset = ImageDataset(args=config)
    loader = dataset.train_loader()

    for (X, y) in loader:
        print(X.shape, y.shape)
        # plt.hist(X.flatten().numpy())
        plt.show()
        break