import wandb
import torch
from arguments import prepare_parser
from dataset_dataloader import ImageDataset


# --------------------------------------------------------------------------------------
# Weights & Biases
def get_wandb_runs(entity="fall-2021-vae-claartje-wilker", project="fall-2021-VAE"):
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)
    return runs


# --------------------------------------------------------------------------------------
# Plotting
def make_c_dict(unique_vals):

    color_dict = {'blue': '#8caadc',
                  'red': '#c51914',
                  'pink': '#fcb1ca',
                  'orange': '#efb116',
                  'dark_blue': '#000563',
                  'green': '#005f32',
                  'sand': '#cec3bc'}

    assert len(unique_vals) < len(list(color_dict.keys())), "too many values to make C-dict"
    colors = [c for c in list(color_dict.values())]

    return {u: colors[i] for i, u in enumerate(unique_vals)}


# --------------------------------------------------------------------------------------
# Data
def get_n_data_samples_x_y(image_dataset_name="bmnist", N_samples=500):
    args = prepare_parser(jupyter=True, print_settings=False)
    args.image_dataset_name = image_dataset_name
    dataset = ImageDataset(args=args)

    data_X, data_y = [], []

    # [B, C, W, H] [B]
    for i, (X, y) in enumerate(dataset.valid_loader(num_workers=1, batch_size=100, shuffle=True)):
        data_X.append(X)
        data_y.append(y)

        if (i + 1) * 100 >= N_samples:
            break

    data_X, data_y = torch.cat(data_X, dim=0), torch.cat(data_y, dim=0)

    return data_X[:N_samples], data_y[:N_samples]


def get_test_validation_loader(image_dataset_name="bmnist", batch_size=100, num_workers=3, include_train=False):
    args = prepare_parser(jupyter=True, print_settings=False)

    args.batch_size = batch_size
    args.image_dataset_name = image_dataset_name
    args.num_workers = num_workers

    dataset = ImageDataset(args=args)
    data_loaders = dict(test=dataset.test_loader(shuffle=False),
                        valid=dataset.valid_loader(shuffle=False))
    if include_train:
        data_loaders["train"] = dataset.train_loader(batch_size=batch_size, shuffle=False)

    return data_loaders
