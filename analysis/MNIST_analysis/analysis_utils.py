import wandb
import torch
from arguments import prepare_parser
from dataset_dataloader import ImageDataset, LanguageDataset


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
def get_n_data_samples_x_y(image_dataset_name="bmnist", language_dataset_name="ptb",
                               image_or_language="language", N_samples=500, phase="valid"):

    # if image_or_language == "language":
    #     print("Warning, get_n_data_samples_x_y not implemented for language yet.")

    args = prepare_parser(jupyter=True, print_settings=False)
    args.image_dataset_name = image_dataset_name
    args.language_dataset_name = language_dataset_name

    if image_or_language == "image":
        dataset = ImageDataset(args=args)
    else:
        dataset = LanguageDataset(args=args)

    if phase == "valid":
        loader = dataset.valid_loader(num_workers=1, batch_size=100, shuffle=True)
    elif phase == "train":
        loader = dataset.train_loader(num_workers=1, batch_size=100, shuffle=True)
    else:
        loader = dataset.test_loader(num_workers=1, batch_size=100, shuffle=True)

    if image_or_language == "image":
        data_X, data_y = [], []

        # [B, C, W, H] [B]
        for i, (X, y) in enumerate(loader):
            data_X.append(X)
            data_y.append(y)

            if (i + 1) * 100 >= N_samples:
                break

        data_X, data_y = torch.cat(data_X, dim=0), torch.cat(data_y, dim=0)

        return data_X[:N_samples], data_y[:N_samples]

    else:
        print("Returns concatted input_ids, attention_masks tensors")
        input_ids, attention_masks = [], []
        for i, batch in enumerate(loader):
            input_ids.append(batch["input_ids"])
            attention_masks.append(batch["attention_mask"])

            if (i + 1) * 100 >= N_samples:
                break

        input_ids, attention_masks = torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

        return input_ids[:N_samples], attention_masks[:N_samples]


def get_test_validation_loader(image_dataset_name="bmnist", language_dataset_name="ptb",
                               image_or_language="language", batch_size=100, num_workers=3, include_train=False):
    args = prepare_parser(jupyter=True, print_settings=False)

    args.batch_size = batch_size
    args.language_dataset_name = language_dataset_name
    args.image_dataset_name = image_dataset_name
    args.num_workers = num_workers

    if image_or_language == "image":
        dataset = ImageDataset(args=args)
    else:
        dataset = LanguageDataset(args=args)

    data_loaders = dict(test=dataset.test_loader(shuffle=True),
                        valid=dataset.valid_loader(shuffle=True))
    if include_train:
        data_loaders["train"] = dataset.train_loader(batch_size=batch_size, shuffle=True)

    return data_loaders
