from torchvision import transforms, datasets
import torch



def load_mnist(batch_size, save_to, height=28, width=28):
    """
    :param batch_size: the dataloader will create batches of this size
    :param save_to: a folder where we download the data into
    :param height: using something other than 28 implies a Resize transformation
    :param width: using something other than 28 implies a Resize transformation
    :return: 3 data loaders
        training, validation, test
    """
    # create directory
    pathlib.Path(save_to).mkdir(parents=True, exist_ok=True)

    if height == width == 28:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()]
        )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            datasets.MNIST(
                save_to,
                train=True,
                download=True,
                transform=transform),
            indices=range(55000)),
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            datasets.MNIST(
                save_to,
                train=True,
                download=True,
                transform=transform),
            indices=range(55000, 60000)),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            save_to,
            train=False,
            download=True,
            transform=transform),
        batch_size=batch_size
    )
    return train_loader, valid_loader, test_loader