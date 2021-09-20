import wandb
import torch


def init_logging(vae_model, args):
    """Initialise W&B logging."""
    print("W&B INIT: RUN NAME", args.run_name)
    wandb.init(project=args.wandb_project, name=args.run_name, entity='fall-2021-vae-claartje-wilker', config=args)
    # wandb.init(name=args.run_name, project=args.wandb_project, config=args)
    # wandb.watch(vae_model) this gives an error on LISA

def log_mog(vae_model, args):
    if args.p_z_type == "mog":
        mix = vae_model.gen_model.mix_components.data  # [n_comp]
        mean = vae_model.gen_model.component_means.data  # [n_comp, D]
        scale = vae_model.gen_model.component_scales.data # [n_comp, D]



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
        wandb_log_dict[f"{phase}/{k}"] = v

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
