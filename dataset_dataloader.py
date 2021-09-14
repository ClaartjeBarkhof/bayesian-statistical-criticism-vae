# Claartje: parts of this code are taken from Wilker's code:
# https://github.com/probabll/mixed-rv-vae/blob/master/data.py

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


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

        assert_data_arguments(args)

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


def assert_data_arguments(args):
    valid_dists = ["multinomial", "bernoulli"]
    assert args.data_distribution in valid_dists, \
        f"Invalid data distribution: {args.data_distribution}, must be one of: {valid_dists}"
    assert not (args.image_dataset_name == "bmnist" and args.data_distribution == "multinomial"), \
        f"If the data set is Binarised MNIST, the data distribution should be set to bernoulli, " \
        f"currently set to {args.data_distribution}."
    assert not (args.image_dataset_name in ["fmnist", "mnist"] and args.data_distribution == "bernoulli"), \
        f"If the data set is MNIST or Fashion MNIST, the data distribution should be set to " \
        f"multinomial, currently set to {args.data_distribution}."
    assert not (args.image_dataset_name in ["bminst", "fmnist", "mnist"] and not args.n_channels == 1), \
        f"{args.image_dataset_name} is a 1-channel dataset."


def greater_than_zero(x):
    return x > 0


def to_float(x):
    return x.float()
