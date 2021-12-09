# Claartje: parts of this code are taken from Wilker's code:
# https://github.com/probabll/mixed-rv-vae/blob/master/data.py

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import torch.utils.data as data_utils

from datasets import load_dataset, list_datasets, load_from_disk, ReadInstruction  # type: ignore
from transformers import RobertaTokenizerFast  # type: ignore

import os


class LanguageDataset:
    def __init__(self, args):

        assert args.language_dataset_name in ["yahoo_answer", "ptb"], "only support 'yahoo_answer' dataset for now"

        self.dataset_name = args.language_dataset_name
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.tokenizer_name = args.tokenizer_name
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.tokenizer_name)

        data_path = f"/home/cbarkhof/fall-2021/data/{self.dataset_name}-{self.tokenizer_name}-seqlen-{self.max_seq_len}"

        self.datasets = dict()
        self.encoded_datasets = dict()

        if os.path.exists(data_path):
            print("Is file!")
            for split in ['train', 'validation', 'test']:
                self.datasets[split] = load_from_disk(data_path + "/" + split)
                print(split, len(self.datasets[split]))
        else:
            print("New pre-processing")
            if self.dataset_name == "ptb":
                for split in ['train', 'validation', 'test']:
                    self.datasets[split] = load_dataset("ptb_text_only", ignore_verifications=True, split=split)

            elif self.dataset_name == "yahoo_answer":
                self.datasets["train"] = load_dataset("yahoo_answers_topics",
                                                      split=ReadInstruction('train', from_=0, to=10, unit='%'))
                self.datasets["validation"] = load_dataset("yahoo_answers_topics",
                                                           split=ReadInstruction('test', from_=0, to=10, unit='%'))
                self.datasets["test"] = load_dataset("yahoo_answers_topics",
                                                     split=ReadInstruction('test', from_=10, to=20, unit='%'))

            for split in ['train', 'validation', 'test']:
                self.datasets[split] = self.datasets[split].map(self.convert_to_features, batched=True)
                columns = ['attention_mask', 'input_ids']

                print("--> data_path", data_path)

                self.datasets[split].set_format(type='torch', columns=columns)

                self.datasets[split].save_to_disk(data_path + "/" + split)

                print(f"Saved split {split} in {data_path + '/' + split}")

    def get_data_loader(self, phase, shuffle=True, batch_size=None, num_workers=None):
        data_loader = DataLoader(self.datasets[phase], collate_fn=self.collate_fn, shuffle=shuffle,
                                 batch_size=self.batch_size if batch_size is None else batch_size,
                                 num_workers=self.num_workers if num_workers is None else num_workers,
                                 pin_memory=self.pin_memory)
        return data_loader

    def train_loader(self, shuffle=True, batch_size=None, num_workers=None):
        return self.get_data_loader("train", shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    def valid_loader(self, num_workers=None, batch_size=None, shuffle=False):
        return self.get_data_loader("validation", shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    def test_loader(self, batch_size=None, shuffle=False, num_workers=None):
        return self.get_data_loader("test", shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    def get_train_validation_loaders(self):
        loaders = dict(train=self.train_loader(), valid=self.valid_loader())
        return loaders

    def collate_fn(self, examples):
        """
        A function that assembles a batch. This is where padding is done, since it depends on
        the maximum sequence length in the batch.

        :param examples: list of truncated, tokenised & encoded sequences
        :return: padded_batch (batch x max_seq_len)
        """

        # Get rid of text and label data, just
        examples = [{"attention_mask": e["attention_mask"], "input_ids": e["input_ids"]} for e in examples]

        # Combine the tensors into a padded batch
        padded_batch = self.tokenizer.pad(examples, return_tensors='pt', padding="max_length",
                                          max_length=self.max_seq_len,
                                          return_attention_mask=True)

        return padded_batch

    def convert_to_features(self, data_batch):
        """
        Truncates and tokenises & encodes a batch of text samples.

        ->  Note: does not pad yet, this will be done in the DataLoader to allow flexible
            padding according to the longest sequence in the batch.

        :param data_batch: batch of text samples
        :return: encoded_batch: batch of samples with the encodings with the defined tokenizer added
        """

        if self.dataset_name == "yahoo_answer":
            key = "best_answer"
        elif self.dataset_name == "ptb":
            key = "sentence"
        else:
            raise NotImplementedError

        print("convert to features")
        encoded_batch = self.tokenizer(data_batch[key], truncation=True, max_length=self.max_seq_len)

        return encoded_batch


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


if __name__ == "__main__":
    from arguments import prepare_parser
    # import matplotlib.pyplot as plt
    #
    config = prepare_parser(jupyter=False, print_settings=True)
    config.language_dataset_name = "yahoo_answer"
    config.image_or_language = "language"
    # dataset = ImageDataset(args=config)
    # loader = dataset.train_loader()
    #
    # for (X, y) in loader:
    #     print(X.shape, y.shape)
    #     # plt.hist(X.flatten().numpy())
    #     plt.show()
    #     break

    if config.image_or_language == "language":

        dataset = LanguageDataset(args=config)
        train_loader = dataset.train_loader(shuffle=True, batch_size=8, num_workers=1)

        for batch in train_loader:
            masks = batch["attention_mask"]
            inputs = batch["input_ids"]
            print("batch inputs", inputs.shape)
            print("batch masks", masks.shape)
            break
