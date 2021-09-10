from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

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

class ImageDataset:
    def __init__(self, args):

        self.image_dataset_name = args.image_dataset_name
        self.image_w = args.image_w
        self.image_h = args.image_h
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size

        # TRANSFORMS
        if self.image_h == self.image_w == 28:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_h, self.image_w)),
                transforms.ToTensor()]
            )

    def train_loader(self):
        train_loader = DataLoader(Subset(datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=self.transform),
                indices=range(55000)),
            batch_size=self.batch_size,
            shuffle=True)
        return train_loader

    def valid_loader(self):
        valid_loader = DataLoader(Subset(datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=self.transform),
                indices=range(55000, 60000)),
            batch_size=self.batch_size,
            shuffle=True)
        return valid_loader

    def test_loader(self):
        test_loader = DataLoader(datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=self.transform),
            batch_size=self.batch_size)
        return test_loader


if __name__ == "__main__":
    # Test code
