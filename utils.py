import wandb
import collections
from tabulate import tabulate
from vae_model.vae import VaeModel
from arguments import prepare_parser
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import torch
import torch.nn.functional as F


def init_logging(vae_model, args):
    """Initialise W&B logging."""
    print("W&B INIT: RUN NAME", args.run_name)
    wandb.init(project=args.wandb_project, dir=args.wandb_dir, name=args.run_name, entity='fall-2021-vae-claartje-wilker', config=args)
    # Define the custom x axis metric

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

    if args.decoder_MADE_gating and args.decoder_network_type == "conditional_made_decoder":
        # h hidden layers
        h = len(vae_model.gen_model.decoder_network.made.hidden_sizes)
        for i in range(h):
            wandb.define_metric(f"layer_{i}_avg_gate_value", step_metric="epoch")
            wandb.define_metric(f"layer_{i}_std_gate_value", step_metric="epoch")

    # wandb.init(name=args.run_name, project=args.wandb_project, config=args)
    # wandb.watch(vae_model) this gives an error on LISA


# GET IMSHOW DATA
def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
    return d


def log_mog(vae_model, args, epoch, log=True, plot=False, title=None, save=None):
    if args.p_z_type == "mog":
        print("Plotting Mixture of Gaussians")

        mix = torch.nn.functional.softmax(vae_model.gen_model.mix_components.data, dim=-1)  # [n_comp]
        means = vae_model.gen_model.component_means.data  # [n_comp, D]
        scales = torch.nn.functional.softplus(vae_model.gen_model.component_pre_scales.data) # [n_comp, D]

        mog_n_components, D = means.shape

        # LINSPACE
        lin_space_ticks = 100
        min_lin, max_lin = -5, 5
        x = np.linspace(min_lin, max_lin, lin_space_ticks)

        ys = []
        for d in range(D):
            means_d = means[:, d]
            scales_d = scales[:, d]
            y = mix_pdf(x.tolist(), means_d.tolist(), scales_d.tolist(), mix.tolist())
            ys.append(y)
        ys = np.stack(ys, axis=0)

        # PLOT
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(ys, aspect=5.0) #, vmin=0.0, vmax=4.5

        # X TICKS
        n_ticks = 10
        step_size = lin_space_ticks // n_ticks
        locs = np.arange(0, lin_space_ticks, step_size)
        labels = [f"{a:.2f}" for a in x[::step_size]]
        plt.xticks(locs, labels)

        # Y TICKS
        ax.set_yticks(np.arange(0, D, 1))
        ax.set_yticklabels(np.arange(1, D + 1, 1))
        ax.set_yticks(np.arange(-.5, D, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        if title is not None:
            plt.title(title)
        else:
            plt.title(f"MoG plot, end of epoch {epoch}")
        plt.xlabel("z")
        plt.ylabel("dim")
        plt.colorbar(im)

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches="tight")

        if log:
            wandb.log({"MoG plot (end of epoch)": plt, "epoch": epoch})

        if plot:
            plt.show()


def log_gates(vae_model, args, epoch):
    if args.decoder_MADE_gating and args.decoder_network_type == "conditional_made_decoder" \
            and args.decoder_MADE_gating_mechanism == 0:
        gate_dict = dict(epoch=epoch)

        made = vae_model.gen_model.decoder_network.made

        for i in range(len(made.hidden_sizes)):
            gate_values = made.__getattr__(f"gate_h_{i}").data
            avg_gate_values = F.sigmoid(gate_values).mean().item()
            std_gate_values = F.sigmoid(gate_values).std().item()
            gate_dict[f"layer_{i}_avg_gate_value"] = avg_gate_values
            gate_dict[f"layer_{i}_std_gate_value"] = std_gate_values

        wandb.log(gate_dict)


def insert_epoch_stats(epoch_stats, loss_dict):
    stat_dict = clean_loss_dict_log_print(loss_dict)

    for k, v in stat_dict.items():
        if k not in epoch_stats:
            epoch_stats[k] = [v]
        else:
            epoch_stats[k].append(v)

    return epoch_stats


def reduce_and_log_epoch_stats(epoch_stats, phase, epoch, step, print_stats=True, log_stats=False):
    print_list = []
    mean_reduced = dict()
    wandb_log_dict = {}

    for i, (k, v) in enumerate(epoch_stats.items()):
        mean, std = np.mean(v), np.std(v)
        mean_reduced[k] = mean
        wandb_log_dict[f"{phase}_epoch/{k} std"] = std
        wandb_log_dict[f"{phase}_epoch/{k} mean"] = mean
        print_list.append([i, k, f"{mean:.2f}", f"{std:.2f}"])

    wandb_log_dict["epoch"] = epoch
    wandb_log_dict["global step"] = step

    if log_stats:
        wandb.log(wandb_log_dict)

    if print_stats:
        print()
        print("---------------------------------------------")
        print(f"** End of epoch {epoch}, phase {phase}, train step {step}")
        print(tabulate(print_list, headers=["", "Metric", "Epoch mean", "Epoch std."]))
        print("---------------------------------------------")

    return mean_reduced


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
        wandb_log_dict[f"{phase}_batch/{k}"] = v

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


def make_checkpoint(model, args, optimisers, epoch, step, mean_reduced_epoch_stats):
    c_path = f"{args.checkpoint_dir}{args.run_name}.pt"

    print()
    print("*"*40)
    best_val_loss = mean_reduced_epoch_stats["total_loss"]
    print(f"Found new best validation loss {best_val_loss:.2f}")
    print("*" * 40)
    print()

    state = {
        'best_val_loss': best_val_loss,
        'mean_epoch_stats': mean_reduced_epoch_stats,
        'epoch': epoch,
        'args': args,
        'step': step,
        'state_dict': model.state_dict(),
    }

    for opt_name, opt in optimisers.items():
        state[opt_name] = opt.state_dict()

    torch.save(state, c_path)


def load_checkpoint_model_for_eval(checkpoint_path, map_location="cuda:0"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if "args" in checkpoint:
        args = checkpoint["args"]
    else:
        args = prepare_parser(print_settings=False, jupyter=True)
        args.latent_dim = 10
        args.encoder_network_type = checkpoint_path.split(" ")[1]
        args.q_z_x_type = checkpoint_path.split(" ")[3]
        args.decoder_network_type = checkpoint_path.split(" ")[6]

    vae_model = VaeModel(args=args)
    vae_model.load_state_dict(checkpoint["state_dict"])
    vae_model.eval()

    if "mean_epoch_stats" in checkpoint:
        p_str = ""
        for k, v in checkpoint["mean_epoch_stats"].items():
            p_str += f"{k}: {v:.2f} | "
        print(p_str)
    else:
        l = checkpoint["best_val_loss"]
        print(f"best val loss {l:.2f}")

    return vae_model

