import os
import pathlib

import torch
import torch.distributions as td
from torch.utils.data import TensorDataset, DataLoader
from vae_model.distributions import AutoRegressiveDistribution

if "cbarkhof" in str(pathlib.Path(__file__).parent.resolve()):
    CODE_DIR = "/home/cbarkhof/fall-2021"
else:
    CODE_DIR = "/Users/claartje/Dropbox/Werk/2021 ILLC/marginal-kl-vae"

# import sys

# sys.path.append(CODE_DIR)

import numpy as np
import pickle

from train import test_from_checkpoint
from utils import load_checkpoint_model_for_eval
from analysis.analysis_utils import get_wandb_runs, get_test_validation_loader

import pandas as pd

pd.options.display.float_format = '{:.6f}'.format

import wandb

import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
import matplotlib.patches as mpatches
from adjustText import adjust_text

PLOT_DIR = f"{CODE_DIR}/notebooks/plots"
ANALYSIS_DIR = f"{CODE_DIR}/analysis/analysis-files"
CHECKPOINT_DIR = f"{CODE_DIR}/run_files/checkpoints"

ENCODE_RECONSTUCT_FILE = f"encode-reconstruct-test-valid.pt"
SAMPLE_FILE = f"generative-samples.pt"

SURPRISAL_RECONSTRUCT_FILE = "surprisal_reconstruct.pt"
SURPRISAL_SAMPLE_FILE = "surprisal_sample.pt"
SURPRISAL_DATA_FILE = "surprisal_data.pt"


# -------------------------------------------------------------------------------------------------------------------
# STEP 0: Fetch run data from W&B based on and filter on run_name prefixes
def make_run_overview_df(prefixes):
    runs = get_wandb_runs(entity="fall-2021-vae-claartje-wilker", project="fall-2021-VAE")

    q_z_x_mapper = {
        "conditional_gaussian_made": "MADE",
        "independent_gaussian": "ind. Gauss."
    }

    decoder_mapper = {
        "conditional_made_decoder": "MADE",
        "cond_pixel_cnn_pp": "PixelCNN++",
        "basic_deconv_decoder": "CNN.T",
        "basic_mlp_decoder": "MLP"
    }

    p_z_mapper = {
        "isotropic_gaussian": "standard",
        "mog": "MoG"
    }

    sum_stats = dict()

    for run in runs:

        for pf in prefixes:
            if pf in run.name:
                obj = run.config["objective"]

                l_rate, l_mmd, beta, mdr_val, fb = 0, 0, 0, 0, 0

                dec = run.config["decoder_network_type"]

                if obj == "INFO-VAE":
                    l_rate = run.config["info_lambda_1_rate"]
                    l_mmd = run.config["info_lambda_2_mmd"]
                    clean_name = f"INFO-VAE l_Rate {l_rate} l_MMD {l_mmd} dec: {decoder_mapper[dec]}"
                elif obj == "BETA-VAE":
                    beta = run.config["beta_beta"]
                    clean_name = f"BETA-VAE beta {beta} dec: {decoder_mapper[dec]}"
                elif obj == "MDR-VAE":
                    mdr_val = run.config["mdr_value"]
                    clean_name = f"MDR-VAE {mdr_val} dec: {decoder_mapper[dec]}"
                elif obj == "FB-VAE":
                    fb = run.config["free_bits"]
                    clean_name = f"FB-VAE {fb} dec: {decoder_mapper[dec]}"
                else:
                    print(f"make_run_overview_df: No clean name builder for this objective: {obj}")
                    continue

                d = {
                    "objective": obj,
                    "l_rate": l_rate,
                    "beta_beta": beta,
                    "free_bits": fb,
                    "mdr_value": mdr_val,
                    "l_mmd": l_mmd,
                    "decoder": dec,
                    "run_name": run.name,
                }

                sum_stats[clean_name] = d

                break

    df = pd.DataFrame(sum_stats).transpose()

    return df


# -------------------------------------------------------------------------------------------------------------------
# STEP 1: Save data to analyse: encode, reconstruct and sample from the models
def encode_reconstruct(data_loaders, vae_model, device, encode_reconstruct_file, include_train=True):
    phases = ["train", "valid", "test"] if include_train else ["test", "valid"]
    encode_reconstruct_d = {phase: dict(z=[], mean=[], scale=[], reconstruct=[]) for phase in phases}

    for phase, data_loader in data_loaders.items():
        for i, (x_in, y) in enumerate(data_loader):
            print(f"{phase} {i}/{len(data_loader)}", end="\r")

            x_in = x_in.to(device)

            # Do a encode and reconstruct
            # Infer the approximate posterior q(z|x) and sample from it to obtain z_post
            # [B, D], [S, B, D]
            q_z_x, z_post = vae_model.inf_model(x_in=x_in, n_samples=1)

            if isinstance(q_z_x, td.Independent):
                mean, scale = q_z_x.base_dist.loc, q_z_x.base_dist.scale

            elif isinstance(q_z_x, AutoRegressiveDistribution):
                (mean, scale) = q_z_x.params

            # Normal normal
            else:
                mean, scale = q_z_x.loc, q_z_x.scale

            # mean, scale: B, D
            # z_post: S, B, D = 1, B, D

            encode_reconstruct_d[phase]["z"].append(z_post.squeeze(0))
            encode_reconstruct_d[phase]["mean"].append(mean)
            encode_reconstruct_d[phase]["scale"].append(scale)

            # Make predictions / generate based on the inferred latent
            reconstruct_x = vae_model.gen_model(x_in=x_in, z_post=z_post).sample().squeeze(0)

            encode_reconstruct_d[phase]["reconstruct"].append(reconstruct_x)

    # Cat
    for phase in ["test", "valid"]:
        for key in ["z", "mean", "scale", "reconstruct"]:
            encode_reconstruct_d[phase][key] = torch.cat(encode_reconstruct_d[phase][key], dim=0).cpu()

    # for phase in ["test", "valid"]:
    #     print(phase)
    #     for k, v in encode_reconstruct_d[phase].items():
    #         print(k, v.shape)

    print("Saving encodings!")
    torch.save(encode_reconstruct_d, encode_reconstruct_file)


def sample(vae_model, n_sample_batches, device, sample_batch_size, sample_file):
    samples = dict(z=[], x=[])
    for i in range(n_sample_batches):
        print(f"sample batch {i}/{n_sample_batches}", end="\r")

        sample_z, sample_x = vae_model.gen_model.sample_generative_model(Sx=1, Sz=sample_batch_size,
                                                                         return_z=True, device=device)
        sample_x = sample_x.squeeze(0).squeeze(1)
        sample_z = sample_z.squeeze(1)

        samples["z"].append(sample_z)
        samples["x"].append(sample_x)

    for key in ["z", "x"]:
        samples[key] = torch.cat(samples[key], dim=0).cpu()

    # for k, v in samples.items():
    #     print(k, v.shape)

    print("Saving samples!")
    torch.save(samples, sample_file)


def encode_reconstruct_sample(df, device="cuda:0", include_train=True, n_sample_batches=20, sample_batch_size=100):
    assert "run_name" in df.columns, "encode_reconstruct_sample: the DF must have a run_name column"

    data_loaders = get_test_validation_loader(image_dataset_name="bmnist",
                                              batch_size=100, num_workers=3, include_train=include_train)

    for i, run_name in enumerate(df["run_name"].values):
        print(i, run_name)

        # TODO: fix this
        if "cond_pixel_cnn_pp" in run_name:
            continue

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        encode_reconstruct_file = f"{save_dir}/{ENCODE_RECONSTUCT_FILE}"
        sample_file = f"{save_dir}/{SAMPLE_FILE}"

        if os.path.isfile(encode_reconstruct_file) and os.path.isfile(sample_file):
            print("Did this run already, continuing!")
            continue

        checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
        vae_model = load_checkpoint_model_for_eval(checkpoint_path, map_location="cuda:0", return_args=False).to(device)

        # ENCODE / RECONSTRUCT
        if not os.path.isfile(encode_reconstruct_file):
            with torch.no_grad():
                encode_reconstruct(data_loaders=data_loaders, vae_model=vae_model, device=device,
                                   encode_reconstruct_file=encode_reconstruct_file, include_train=include_train)

        # SAMPLE
        if not os.path.isfile(sample_file):
            with torch.no_grad():
                sample(vae_model=vae_model, n_sample_batches=n_sample_batches, device=device,
                       sample_batch_size=sample_batch_size, sample_file=sample_file)


# -------------------------------------------------------------------------------------------------------------------
# STEP 2: Gather surprisal stats for data, reconstructions and model samples
def surprisal_reconstructions(reconstruct_file, surprisal_reconstruct_file, vae_model,
                              batch_size_surprisal, n_iw_samples, device):
    print("Conditional generation - reconstructions")

    surprisal_reconstruct = dict()
    rec_dict = torch.load(reconstruct_file)

    for phase in ['valid', 'test', 'train']:
        # [N, 1, 28, 28]
        reconstruct_loader = DataLoader(TensorDataset(rec_dict[phase]["reconstruct"]),
                                        batch_size=batch_size_surprisal)
        iw_ll_recon = []
        for i, x_in in enumerate(reconstruct_loader):
            print(f"... calculating importance weighted (n_samples={n_iw_samples})"
                  f" log likelihood {i:3d}/{len(reconstruct_loader)}", end="\r")

            x_in = x_in[0].to(device)

            iw_ll = vae_model.estimate_log_likelihood_batch(x_in, n_samples=n_iw_samples, per_bit=False)
            iw_ll_recon.append(iw_ll)

        surprisal_reconstruct[phase] = torch.cat(iw_ll_recon).cpu()
        print("reconstruct", phase, "shape", surprisal_reconstruct[phase].shape)

    print("Saving surprisal stats reconstructions")
    torch.save(surprisal_reconstruct, surprisal_reconstruct_file)

def surprisal_data(surprisal_data_file, data_loaders, vae_model, n_iw_samples):
    print("Data")
    for phase, data_loader in data_loaders.items():
        surprisal_data[phase] = torch.Tensor(
            vae_model.estimate_log_likelihood_dataset(data_loader, n_samples=n_iw_samples))
        print("data", phase, "shape", surprisal_data[phase].shape)
    print("Saving surprisal stats data")
    torch.save(surprisal_data, surprisal_data_file)


def surprisal_samples(sample_file, surprisal_sample_file, vae_model, batch_size_surprisal, n_iw_samples, device):
    print("Unconditional generation - samples")
    sample_dict = torch.load(sample_file)

    # [N, 1, 28, 28]
    sample_loader = DataLoader(TensorDataset(sample_dict["x"]),
                               batch_size=batch_size_surprisal)

    iw_ll_sample = []
    for i, x_in in enumerate(sample_loader):
        print(f"... calculating importance weighted (n_samples={n_iw_samples})"
              f" log likelihood {i:3d}/{len(sample_loader)}", end="\r")

        x_in = x_in[0].to(device)

        iw_ll = vae_model.estimate_log_likelihood_batch(x_in, n_samples=n_iw_samples, per_bit=False)
        iw_ll_sample.append(iw_ll)

    surprisal_samples = torch.cat(iw_ll_sample).cpu()

    print("samples shape", surprisal_samples.shape)

    print("Saving surprisal stats samples")
    torch.save(surprisal_samples, surprisal_sample_file)


def gather_surprisal_stats(device="cuda:0", include_train=True, batch_size_surprisal=100, n_iw_samples=50):
    data_loaders = get_test_validation_loader(image_dataset_name="bmnist",
                                              batch_size=100, num_workers=3, include_train=include_train)

    for i, run_name in enumerate(os.listdir(ANALYSIS_DIR)):
        print(i, run_name)

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        reconstruct_file = f"{save_dir}/{ENCODE_RECONSTUCT_FILE}"
        sample_file = f"{save_dir}/{SAMPLE_FILE}"

        surprisal_reconstruct_file = f"{save_dir}/{SURPRISAL_RECONSTRUCT_FILE}"
        surprisal_sample_file = f"{save_dir}/{SURPRISAL_SAMPLE_FILE}"
        surprisal_data_file = f"{save_dir}/{SURPRISAL_DATA_FILE}"

        checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
        vae_model, args = load_checkpoint_model_for_eval(checkpoint_path, map_location=device, return_args=True)
        vae_model = vae_model.to(device)

        if not (os.path.isfile(reconstruct_file) and os.path.isfile(sample_file)):
            print("no encoding / sample files, skipping for now. run the sample-encode-reconstruct NB first.")
            continue

        # Conditional generation - reconstructions
        if not os.path.isfile(surprisal_reconstruct_file):
            with torch.no_grad():
                surprisal_reconstructions(reconstruct_file=reconstruct_file,
                                          surprisal_reconstruct_file=surprisal_reconstruct_file,
                                          vae_model=vae_model,
                                          batch_size_surprisal=batch_size_surprisal,
                                          n_iw_samples=n_iw_samples, device=device)
        else:
            print("did reconstructions already, skipping!")

        # Unconditional generation - samples
        if not os.path.isfile(surprisal_sample_file):
            with torch.no_grad():
                surprisal_samples(sample_file=sample_file, surprisal_sample_file=surprisal_sample_file,
                                  vae_model=vae_model, batch_size_surprisal=batch_size_surprisal,
                                  n_iw_samples=n_iw_samples, device=device)
        else:
            print("did samples already, skipping!")

        # Data
        if not os.path.isfile(surprisal_data_file):
            with torch.no_grad():
                surprisal_data(surprisal_data_file=surprisal_data_file, data_loaders=data_loaders,
                               vae_model=vae_model, n_iw_samples=n_iw_samples)
        else:
            print("did data already, skipping!")

