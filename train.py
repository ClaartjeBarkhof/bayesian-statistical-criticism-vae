from arguments import preprare_parser
from dataset_dataloader import ImageDataset # LanguageDataset

def get_loaders(config):
    dataset = None
    if config.image_or_language == "image":
        dataset = ImageDataset(image_dataset_name=config.image_dataset_name, image_w=config.image_w,
                               image_h=config.image_h, data_dir=config.data_dir, batch_size=config.batch_size)
    # TODO: else:
    #     dataset = LanguageDataset(...)

    loaders = dict(train=dataset.train_loader, valid=dataset.valid_loader)
    return loaders


def train():
    config = preprare_parser(print_settings=True)


    train_loader =

if __name__ == "__main__":
    train()