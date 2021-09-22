import wandb
import torch
import collections
import numpy as np
import datetime
from tabulate import tabulate

def init_logging(vae_model, args):
    """Initialise W&B logging."""
    print("W&B INIT: RUN NAME", args.run_name)
    wandb.init(project=args.wandb_project, name=args.run_name, entity='fall-2021-vae-claartje-wilker', config=args)
    # Define the custom x axis metric

    # total_loss = total_loss,
    # mmd = mmd,
    # elbo = elbo,
    # mdr_loss = mdr_loss,
    # mdr_multiplier = mdr_multiplier,
    # distortion = distortion,
    # kl_prior_post = kl_prior_post

    phases = ["train", "valid"]
    metrics = ["total_loss", "mmd", "elbo", "mdr_loss", "mdr_multiplier", "distortion", "kl_prior_post", "iw_ll"]

    # Plot all batch metrics against global step
    wandb.define_metric("global step")
    for phase in phases:
        for metric in metrics:
            wandb.define_metric(f"{phase}_batch/{metric}", step_metric='global step')

    # Plot all epoch metrics against epoch
    wandb.define_metric("epoch")
    for phase in phases:
        for metric in metrics:
            wandb.define_metric(f"{phase}_epoch/{metric} std", step_metric='epoch')
            wandb.define_metric(f"{phase}_epoch/{metric} mean", step_metric='epoch')

    # wandb.init(name=args.run_name, project=args.wandb_project, config=args)
    # wandb.watch(vae_model) this gives an error on LISA


# def log_mog(vae_model, args):
#     if args.p_z_type == "mog":
#         mix = vae_model.gen_model.mix_components.data  # [n_comp]
#         mean = vae_model.gen_model.component_means.data  # [n_comp, D]
#         scale = vae_model.gen_model.component_scales.data # [n_comp, D]

def insert_epoch_stats(epoch_stats, loss_dict):
    stat_dict = clean_loss_dict_log_print(loss_dict)

    for k, v in loss_dict.items():
        if k not in epoch_stats:
            epoch_stats[k] = [v]
        else:
            epoch_stats[k].append(v)

    return epoch_stats


def reduce_and_log_epoch_stats(epoch_stats, phase, epoch, step, print_stats=True):
    print_list = []
    wandb_log_dict = {}

    for i, (k, v) in enumerate(epoch_stats.items()):
        mean, std = np.mean(v), np.std(v)
        wandb_log_dict[f"{phase}_epoch/{k} std"] = std
        wandb_log_dict[f"{phase}_epoch/{k} mean"] = mean
        print_list.append([i, k, f"{mean:.2f}", f"{std:.2f}"])

    wandb_log_dict["epoch"] = epoch
    wandb_log_dict["global step"] = step
    wandb.log(wandb_log_dict)

    if print_stats:
        print(f"End of epoch {epoch}, phase {phase}, train step {step}\n")
        print(tabulate(print_list, headers=["", "Metric", "Epoch mean", "Epoch std."]))

    return wandb_log_dict


def make_nested_dict():
    """
    A function to initialise a nested default dict.
    """
    return collections.defaultdict(make_nested_dict)


def clean_loss_dict_log_print(loss_dict):
    log_dict = dict()

    for k, v in loss_dict.items():
        if v is None:
            continue
        else:

            if type(v) == float:
                log_val = v
            else:
                log_val = v.item()

            log_dict[k] = log_val
    return log_dict


def log_step(loss_dict, step, epoch, phase):
    log_dict = clean_loss_dict_log_print(loss_dict)
    wandb_log_dict = dict()
    for k, v in log_dict.items():
        wandb_log_dict[f"{phase}_step/{k}"] = v

    wandb_log_dict["epoch"] = epoch
    wandb_log_dict["global step"] = step
    wandb.log(wandb_log_dict)


def print_step(step, epoch, batch_idx, n_batches, phase, args, loss_dict):
    print_string = f"{phase} | Epoch {epoch}/{args.max_epochs} | Global step {step}/{args.max_steps} | Batch {batch_idx}/{n_batches} |"
    for k, v in clean_loss_dict_log_print(loss_dict).items():
        print_string += f" {k}: {v:.2f}"
    print(print_string)


def determine_device(args):
    device_name = "cpu"

    if args.gpus > 0:
        if not torch.cuda.is_available():
            print("args.gpus > 0, but CUDA not available. Quitting.")
        else:
            if args.gpus > 1 or args.ddp:
                raise NotImplementedError
            else:
                device_name = "cuda:0"

    return device_name


def make_checkpoint(model, args, optimisers, epoch, step, best_val_loss):
    state = {
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'step': step,
        'state_dict': model.state_dict(),
    }

    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    date, time = datetime_stamp.split("--")[0], datetime_stamp.split("--")[1]
    date_time = f"{date}-{time}"

    for opt_name, opt in optimisers.items():
        state[opt_name] = opt.state_dict()

    torch.save(state, f"{args.checkpoint_dir}/{args.run_name}_{date_time}.pt")
