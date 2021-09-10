from arguments import preprare_parser
from dataset_dataloader import ImageDataset # LanguageDataset
from vae_model.vae import VaeModel
import pytorch_lightning as pl


def get_data_loaders(args):
    dataset = None
    if args.image_or_language == "image":
        dataset = ImageDataset(args)
    # TODO: else:
    #     dataset = LanguageDataset(...)

    loaders = dict(train=dataset.train_loader(), valid=dataset.valid_loader())

    return loaders


def main():
    args = preprare_parser(print_settings=True)

    data_loaders = get_data_loaders(args)

    vae_model = VaeModel(args=args)

    trainer = pl.Trainer()

    trainer.fit(vae_model, data_loaders["train"], data_loaders["valid"])


if __name__ == "__main__":
    main()