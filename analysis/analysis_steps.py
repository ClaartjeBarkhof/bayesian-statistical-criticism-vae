import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from train import Trainer
from train import test_from_checkpoint
from utils import load_checkpoint_model_for_eval
from analysis.analysis_utils import get_wandb_runs, get_test_validation_loader, get_n_data_samples_x_y

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

AVG_DATA_SAMPLE_FILE = f"{CODE_DIR}/analysis/analysis-files/average_val_data_samples_per_class.p"

SURPRISAL_RECONSTRUCT_FILE = "surprisal_reconstruct.pt"
SURPRISAL_SAMPLE_FILE = "surprisal_sample.pt"
SURPRISAL_DATA_FILE = "surprisal_data.pt"

TEST_VALID_EVAL_FILE = "test-valid-results.pt"

KNN_PREDICT_SAMPLES_FILE = "knn-preds-generative-samples.pickle"
KNN_PREDICT_RECONSTRUCTIONS_FILE = "knn-preds-reconstructions.pickle"

KNN_PREDICT_STATS_FILE = "knn-preds-stats.pickle"

DATA_SPACE_STATS = "data_space_stats.pickle"


# -------------------------------------------------------------------------------------------------------------------
# STEP 0: Fetch run data from W&B based on and filter on run_name prefixes
def make_run_overview_df(prefixes, add_data_group=False):
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

    if add_data_group:
        data_group = {
            "objective": "data_group",
            "l_rate": 0,
            "beta_beta": 0,
            "free_bits": 0,
            "mdr_value": 0,
            "l_mmd": 0,
            "decoder": "data_group",
            "run_name": "data_group"
        }
        df = df.append(data_group, ignore_index=True)

    return df


def overview_of_missing_analysis(df):
    all_missing = dict()

    missing = dict(dir=True, surprisal_recon=True, surprisal_data=True, surprisal_samples=True,
                   encode=True, samples=True, test_valid_eval=True, knn_predict_samples=True,
                   knn_predict_recons=True, knn_predict_stats=True, data_space_stats=True)

    for run_name in df["run_name"].values:
        save_dir = f"{ANALYSIS_DIR}/{run_name}"

        # skip things that are not directories
        if os.path.isfile(save_dir):
            continue

        missing_run = missing.copy()

        if os.path.isdir(save_dir):
            missing_run["dir"] = False

        # ENCODE / RECONSTRUCT
        if os.path.isfile(f"{save_dir}/{ENCODE_RECONSTUCT_FILE}"):
            missing_run["encode"] = False
        if os.path.isfile(f"{save_dir}/{SAMPLE_FILE}"):
            missing_run["samples"] = False

        # TEST / VALID SIMPLE EVAL
        if os.path.isfile(f"{save_dir}/{TEST_VALID_EVAL_FILE}"):
            missing_run["test_valid_eval"] = False

        # SURPRISAL
        if os.path.isfile(f"{save_dir}/{SURPRISAL_DATA_FILE}"):
            missing_run["surprisal_data"] = False
        if os.path.isfile(f"{save_dir}/{SURPRISAL_SAMPLE_FILE}"):
            missing_run["surprisal_samples"] = False
        if os.path.isfile(f"{save_dir}/{SURPRISAL_RECONSTRUCT_FILE}"):
            missing_run["surprisal_recon"] = False

        # KNN PREDS
        if os.path.isfile(f"{save_dir}/{KNN_PREDICT_SAMPLES_FILE}"):
            missing_run["knn_predict_samples"] = False
        if os.path.isfile(f"{save_dir}/{KNN_PREDICT_RECONSTRUCTIONS_FILE}"):
            missing_run["knn_predict_recons"] = False
        if os.path.isfile(f"{save_dir}/{KNN_PREDICT_STATS_FILE}"):
            missing_run["knn_predict_stats"] = False

        # DATA SPACE STATS
        if os.path.isfile(f"{save_dir}/{DATA_SPACE_STATS}"):
            missing_run["data_space_stats"] = False

        any_missing = False
        for k, v in missing_run.items():
            if v is True:
                any_missing = True

            if any_missing:
                break

        if any_missing:
            all_missing[run_name] = missing_run

    df_missing = pd.DataFrame(all_missing).transpose()

    return df_missing


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
    for phase in phases:
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


def encode_reconstruct_sample(df, device="cuda:0", include_train=True, n_sample_batches=20,
                              sample_batch_size=100, reverse=False):
    assert "run_name" in df.columns, "encode_reconstruct_sample: the DF must have a run_name column"

    data_loaders = get_test_validation_loader(image_dataset_name="bmnist",
                                              batch_size=100, num_workers=3, include_train=include_train)

    run_names = list(df["run_name"].values)
    if reverse:
        run_names.reverse()

    for i, run_name in enumerate(run_names):
        print(i, run_name)

        # TODO: fix this
        # if "cond_pixel_cnn_pp" in run_name:
        #     continue

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        encode_reconstruct_file = f"{save_dir}/{ENCODE_RECONSTUCT_FILE}"
        sample_file = f"{save_dir}/{SAMPLE_FILE}"

        done_encode, done_sample = False, False

        if os.path.isfile(encode_reconstruct_file):
            if "train" in torch.load(encode_reconstruct_file):
                done_encode = True

        if os.path.isfile(sample_file):
            done_sample = True

        if done_encode and done_sample:
            print("Done this run already, continue")
            continue

        checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
        vae_model = load_checkpoint_model_for_eval(checkpoint_path, map_location="cuda:0", return_args=False).to(device)

        # ENCODE / RECONSTRUCT
        if not done_encode:
            with torch.no_grad():
                encode_reconstruct(data_loaders=data_loaders, vae_model=vae_model, device=device,
                                   encode_reconstruct_file=encode_reconstruct_file, include_train=include_train)

        # SAMPLE
        if not done_sample:
            with torch.no_grad():
                sample(vae_model=vae_model, n_sample_batches=n_sample_batches, device=device,
                       sample_batch_size=sample_batch_size, sample_file=sample_file)


# -------------------------------------------------------------------------------------------------------------------
# STEP 2: Gather surprisal stats for data, reconstructions and model samples
def surprisal_reconstructions(reconstruct_file, surprisal_reconstruct_file, vae_model,
                              batch_size_surprisal, n_iw_samples, device):
    print("Conditional generation - reconstructions")

    print(f"BATCH SIZE {batch_size_surprisal}, N SAMPLES {n_iw_samples}")

    surprisal_reconstruct = dict()
    rec_dict = torch.load(reconstruct_file)

    for phase in ['valid', 'test', 'train']:
        # [N, 1, 28, 28]
        reconstruct_loader = DataLoader(TensorDataset(rec_dict[phase]["reconstruct"]),
                                        batch_size=batch_size_surprisal, shuffle=True)
        iw_ll_recon = []
        counter = 0
        for i, x_in in enumerate(reconstruct_loader):
            print(f"... calculating importance weighted (n_samples={n_iw_samples})"
                  f" log likelihood {i:3d}/{len(reconstruct_loader)}", end="\r")

            x_in = x_in[0].to(device)

            iw_ll = vae_model.estimate_log_likelihood_batch(x_in, n_samples=n_iw_samples, per_bit=False)
            iw_ll_recon.append(iw_ll)
            counter += x_in.shape[0]

            # we don't need it all
            if counter > 2000:
                break

        surprisal_reconstruct[phase] = torch.cat(iw_ll_recon).cpu()
        print("reconstruct", phase, "shape", surprisal_reconstruct[phase].shape)

    print("Saving surprisal stats reconstructions")
    torch.save(surprisal_reconstruct, surprisal_reconstruct_file)


def surprisal_data(surprisal_data_file, data_loaders, vae_model, n_iw_samples, batch_size=30):
    print("Data")
    d = dict()
    for phase, data_loader in data_loaders.items():
        max_batches = int(2000 / batch_size)
        d[phase] = torch.Tensor(
            vae_model.estimate_log_likelihood_dataset(data_loader, max_batches=max_batches, n_samples=n_iw_samples))
        print("data", phase, "shape", d[phase].shape)
    print("Saving surprisal stats data")
    torch.save(d, surprisal_data_file)


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


def gather_surprisal_stats(device="cuda:0", include_train=True, batch_size_surprisal=100, n_iw_samples=50, reverse=False):
    data_loaders = get_test_validation_loader(image_dataset_name="bmnist", include_train=include_train,
                                              batch_size=batch_size_surprisal, num_workers=3)
    run_names = list(os.listdir(ANALYSIS_DIR))

    if reverse:
        run_names.reverse()

    for i, run_name in enumerate(run_names):
        if run_name == "data_group":
            continue
        if os.path.isfile(f"{ANALYSIS_DIR}/{run_name}"):
            continue

        print(i, run_name)
        # if "pixel" in run_name.lower():
        #     print("Skipping pixel cnn models for now.")
        #     continue

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)

        reconstruct_file = f"{save_dir}/{ENCODE_RECONSTUCT_FILE}"
        sample_file = f"{save_dir}/{SAMPLE_FILE}"

        surprisal_reconstruct_file = f"{save_dir}/{SURPRISAL_RECONSTRUCT_FILE}"
        surprisal_sample_file = f"{save_dir}/{SURPRISAL_SAMPLE_FILE}"
        surprisal_data_file = f"{save_dir}/{SURPRISAL_DATA_FILE}"

        checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
        vae_model, args = load_checkpoint_model_for_eval(checkpoint_path, map_location=device, return_args=True)
        vae_model = vae_model.to(device)

        # Check if data is there we need
        if not (os.path.isfile(reconstruct_file) or os.path.isfile(sample_file)):
            print("no encoding / sample files, skipping for now. run the sample-encode-reconstruct NB first.")
            continue

        done_surprisal_recon = False
        if os.path.isfile(surprisal_reconstruct_file):
            if "train" in torch.load(surprisal_reconstruct_file):
                done_surprisal_recon = True

        # Conditional generation - reconstructions
        if not done_surprisal_recon:
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
        done_surprisal_data = False
        if os.path.isfile(surprisal_data_file):
            if "train" in torch.load(surprisal_data_file):
                done_surprisal_data = True

        if not done_surprisal_data:
            with torch.no_grad():
                surprisal_data(surprisal_data_file=surprisal_data_file, data_loaders=data_loaders,
                               vae_model=vae_model, n_iw_samples=n_iw_samples, batch_size=batch_size_surprisal)
        else:
            print("did data already, skipping!")


# -------------------------------------------------------------------------------------------------------------------
# STEP 3: Simple evaluate valid and test set (global statistics)
def simple_evaluate_valid_test(df, device="cuda:0", batch_size=100, num_workers=3):
    assert "run_name" in df.columns, "simple_evaluate_valid_test: the DF must have a run_name column"
    save_metrics = ['mmd', 'elbo', 'distortion', 'kl_prior_post', 'mean_mean', 'std_across_x_mean',
                    'std_across_z_mean', 'mean_scale', 'std_across_x_scale', 'std_across_z_scale']

    data_loaders = get_test_validation_loader(image_dataset_name="bmnist", batch_size=batch_size,
                                              num_workers=num_workers, include_train=False)

    for i, run_name in enumerate(df["run_name"].values):
        print(i, run_name)

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)

        # RESULT FILE
        result_file = f"{save_dir}/{TEST_VALID_EVAL_FILE}"
        results = dict()

        if os.path.isfile(result_file):
            print("Did this one already, skipping it for now.")
            continue

        with torch.no_grad():

            # CHECKPOINT
            checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
            vae_model, args = load_checkpoint_model_for_eval(checkpoint_path, map_location=device, return_args=True)
            vae_model = vae_model.to(device)
            trainer = Trainer(args=args, dataset=None, vae_model=vae_model, device=device)

            # TEST / VALIDATE
            for phase, loader in data_loaders.items():

                results[phase] = dict()
                res = trainer.test(device=device, loader=loader)

                for k, v in res.items():
                    if k in save_metrics:
                        results[phase][k] = v
                        results[phase][k + " mean"] = np.mean(v)

            torch.save(results, result_file)


# -------------------------------------------------------------------------------------------------------------------
# STEP 4: Data space statistics
def make_knn_predictions(knn, loader, knn_mimicker=False, device="cuda:0"):
    if knn_mimicker:
        knn = knn.to(device)
        knn.eval()

    preds = dict(proba=[], preds=[], marg=None)
    for i, x_in in enumerate(loader):
        print(f"{i:3d}/{len(loader)}", end="\r")

        # [B, 784]
        if not knn_mimicker:
            x_in = x_in[0].reshape(-1, 28 * 28).numpy()
            proba = knn.predict_proba(x_in)
        else:
            with torch.no_grad():
                x_in = x_in[0].to(device)
                proba = knn.predict_proba(x_in).cpu().numpy()

        preds["proba"].append(proba)
        preds["preds"].append(np.argmax(proba, axis=1))

    preds["proba"] = np.concatenate(preds["proba"], axis=0)
    preds["preds"] = np.concatenate(preds["preds"], axis=0)
    preds["marg"] = preds["proba"].mean(axis=0)

    print(preds["proba"].shape, preds["preds"].shape, preds["marg"].shape)
    return preds


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        out = self(x)
        return torch.softmax(out, dim=1)

    def predict(self, x):
        return torch.argmax(self.predict_proba(x), dim=-1)


def knn_predictions_for_samples_reconstructions(batch_size=100,
                                                knn_mimicker_path="/home/cbarkhof/fall-2021/notebooks/KNN_mimicking_network.pt",
                                                knn_path=None, device="cuda:0"):
    assert knn_path is not None or knn_mimicker_path is not None, "knn_predictions_for_samples_reconstructions: " \
                                                                  "Either provide knn_path or knn_mimicker_path "
    knn_mimicker = False
    if knn_mimicker_path is not None:
        knn_mimicker = True
        knn = Net()
        knn.load_state_dict(torch.load(knn_mimicker_path))
    else:
        knn = pickle.load(open(knn_path, "rb"))

    for i, run_name in enumerate(os.listdir(ANALYSIS_DIR)):
        if os.path.isfile(f"{ANALYSIS_DIR}/{run_name}"):
            continue
        print(i, run_name)

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        sample_file = f"{save_dir}/{SAMPLE_FILE}"
        recon_file = f"{save_dir}/{ENCODE_RECONSTUCT_FILE}"
        knn_sample_save_file = f"{save_dir}/{KNN_PREDICT_SAMPLES_FILE}"
        knn_recon_save_file = f"{save_dir}/{KNN_PREDICT_RECONSTRUCTIONS_FILE}"

        if not os.path.isfile(sample_file):
            print("No samples of this model, run sample-encode-reconstruct NB, continuing!")
            continue

        # if not os.path.isfile(knn_sample_save_file):
        assert os.path.isfile(sample_file), "no samples for this model, run encode_reconstruct_sample"
        samples = torch.load(sample_file)
        sample_loader = DataLoader(TensorDataset(samples["x"]),
                                   batch_size=batch_size)
        sample_preds = make_knn_predictions(knn, sample_loader, knn_mimicker=knn_mimicker, device=device)
        pickle.dump(sample_preds, open(knn_sample_save_file, "wb"))

        del sample_preds
        del sample_loader

        # if not os.path.isfile(knn_recon_save_file):
        assert os.path.isfile(recon_file), "no reconstructions for this model, run encode_reconstruct_sample"
        recons = torch.load(recon_file)
        preds = dict()
        for phase in ["train", "test", "valid"]:
            r = recons[phase]["reconstruct"]
            loader = DataLoader(TensorDataset(r), batch_size=batch_size)
            preds[phase] = make_knn_predictions(knn, loader, knn_mimicker=knn_mimicker, device=device)

        pickle.dump(preds, open(knn_recon_save_file, "wb"))


def make_knn_predictions_data():
    from dataset_dataloader import ImageDataset
    from arguments import prepare_parser
    import pickle

    args = prepare_parser(jupyter=True, print_settings=False)
    args.batch_size = 200
    args.num_workers = 3

    dataset = ImageDataset(args=args)

    knn_mimicker = Net()
    knn_mimicker_path = "/home/cbarkhof/fall-2021/notebooks/KNN_mimicking_network.pt"
    knn_mimicker.load_state_dict(torch.load(knn_mimicker_path))
    knn_mimicker.to("cuda:0")

    r = dict()

    for phase in ["train", "valid", "test"]:
        print(phase)

        if phase == "train":
            loader = dataset.train_loader(shuffle=False)
        elif phase == "valid":
            loader = dataset.valid_loader(shuffle=False)
        else:
            loader = dataset.test_loader(shuffle=False)

        probas = []
        for batch in loader:
            with torch.no_grad():
                x = batch[0].to("cuda:0")
                probas.append(knn_mimicker.predict_proba(x).cpu())
        probas = torch.cat(probas, dim=0)
        marg = probas.mean(dim=0)
        preds = torch.argmax(probas, dim=-1)

        r[phase] = dict(proba=probas, marg=marg, preds=preds)

    path = "/home/cbarkhof/fall-2021/analysis/analysis-files/KNN_data_probas.p"
    pickle.dump(r, open(path, "wb"))


def knn_prediction_distribution_stats(
        knn_data_pred_path="/home/cbarkhof/fall-2021/analysis/analysis-files/KNN_data_probas.p"):
    knn_data_pred = pickle.load(open(knn_data_pred_path, "rb"))

    uniform = td.Categorical(probs=torch.FloatTensor([0.1 for _ in range(10)]))

    marg_test_data = td.Categorical(probs=torch.FloatTensor(knn_data_pred["test"]["marg"]))
    marg_valid_data = td.Categorical(probs=torch.FloatTensor(knn_data_pred["valid"]["marg"]))
    marg_train_data = td.Categorical(probs=torch.FloatTensor(knn_data_pred["train"]["marg"]))

    marginals_data = dict(test=marg_test_data, valid=marg_valid_data, train=marg_train_data)

    # DATA GROUP
    save_dir = f"{ANALYSIS_DIR}/data_group"
    knn_preds_stats_file = f"{save_dir}/{KNN_PREDICT_STATS_FILE}"

    if not os.path.isfile(knn_preds_stats_file):

        instance_dist_test_data = td.Categorical(probs=torch.FloatTensor(knn_data_pred["test"]["proba"]))
        instance_dist_valid_data = td.Categorical(probs=torch.FloatTensor(knn_data_pred["valid"]["proba"]))
        instance_dist_train_data = td.Categorical(probs=torch.FloatTensor(knn_data_pred["train"]["proba"]))

        train_data_kl_instance_marg = td.kl_divergence(instance_dist_train_data, marg_train_data)
        valid_data_kl_instance_marg = td.kl_divergence(instance_dist_valid_data, marg_valid_data)
        test_data_kl_instance_marg = td.kl_divergence(instance_dist_test_data, marg_test_data)

        train_kl_marg_uniform = td.kl_divergence(marg_train_data, uniform)
        valid_kl_marg_uniform = td.kl_divergence(marg_valid_data, uniform)
        test_kl_marg_uniform = td.kl_divergence(marg_test_data, uniform)

        # this should be zero of course
        train_kl_marg_marg_data = td.kl_divergence(marg_train_data, marg_train_data)
        valid_kl_marg_marg_data = td.kl_divergence(marg_valid_data, marg_valid_data)
        test_kl_marg_marg_data = td.kl_divergence(marg_test_data, marg_test_data)

        res = dict()
        res['train'] = dict(kl_instance_marg_pred=train_data_kl_instance_marg.tolist(),
                            kl_instance_marg_pred_mean=train_data_kl_instance_marg.mean().item(),
                            kl_marg_uniform=train_kl_marg_uniform.tolist(),
                            kl_marg_uniform_mean=train_kl_marg_uniform.mean().item(),
                            kl_marg_marg_data=train_kl_marg_marg_data.tolist(),
                            kl_marg_marg_data_mean=train_kl_marg_marg_data.mean().item())

        res['valid'] = dict(kl_instance_marg_pred=valid_data_kl_instance_marg.tolist(),
                            kl_instance_marg_pred_mean=valid_data_kl_instance_marg.mean().item(),
                            kl_marg_uniform=valid_kl_marg_uniform.tolist(),
                            kl_marg_uniform_mean=valid_kl_marg_uniform.mean().item(),
                            kl_marg_marg_data=valid_kl_marg_marg_data.tolist(),
                            kl_marg_marg_data_mean=valid_kl_marg_marg_data.mean().item())

        res['test'] = dict(kl_instance_marg_pred=test_data_kl_instance_marg.tolist(),
                            kl_instance_marg_pred_mean=test_data_kl_instance_marg.mean().item(),
                            kl_marg_uniform=test_kl_marg_uniform.tolist(),
                            kl_marg_uniform_mean=test_kl_marg_uniform.mean().item(),
                            kl_marg_marg_data=test_kl_marg_marg_data.tolist(),
                            kl_marg_marg_data_mean=test_kl_marg_marg_data.mean().item())

        print("DATAGROUP: Saving KNN stats!")
        pickle.dump(res, open(knn_preds_stats_file, "wb"))

    for i, run_name in enumerate(os.listdir(ANALYSIS_DIR)):
        if os.path.isfile(f"{ANALYSIS_DIR}/{run_name}"):
            continue

        save_dir = f"{ANALYSIS_DIR}/{run_name}"

        if not (os.path.isfile(f"{save_dir}/{KNN_PREDICT_RECONSTRUCTIONS_FILE}") or
                os.path.isfile(f"{save_dir}/{KNN_PREDICT_SAMPLES_FILE}")):
            print("No KNN predictions yet. Run knn_predictions_for_samples_reconstructions")
            print(save_dir)
            continue

        knn_preds_recon = pickle.load(open(f"{save_dir}/{KNN_PREDICT_RECONSTRUCTIONS_FILE}", "rb"))
        knn_preds_samples = pickle.load(open(f"{save_dir}/{KNN_PREDICT_SAMPLES_FILE}", "rb"))

        knn_preds_stats_file = f"{save_dir}/{KNN_PREDICT_STATS_FILE}"

        # if os.path.isfile(knn_preds_stats_file):
        #     print("Did this one already, skipping!")
        #     continue

        res = dict()

        # KNN PREDICTION DISTRIBUTIONS
        # RECONSTRUCTIONS & SAMPLES
        # Some measurements over marginal prediction distribution versus instance prediciton distributions
        # phase: preds, proba, marg
        for phase in ["test", "valid", "samples"]:
            if phase != "samples":
                instance_dists = td.Categorical(probs=torch.FloatTensor(knn_preds_recon[phase]["proba"]))
                marginal_pred = td.Categorical(probs=torch.FloatTensor(knn_preds_recon[phase]["marg"]))
            else:
                instance_dists = td.Categorical(probs=torch.FloatTensor(knn_preds_samples["proba"]))
                marginal_pred = td.Categorical(probs=torch.FloatTensor(knn_preds_samples["marg"]))

            # KL [ p(y|z) | p(y) ]
            kl_instance_marg_pred = td.kl_divergence(instance_dists, marginal_pred)  # .mean().item()
            kl_instance_marg_pred_mean = kl_instance_marg_pred.mean().item()
            kl_marg_uniform = td.kl_divergence(marginal_pred, uniform)  # .mean().item()
            kl_marg_uniform_mean = kl_marg_uniform.mean().item()

            res[phase] = dict(kl_instance_marg_pred=kl_instance_marg_pred.tolist(),
                              kl_instance_marg_pred_mean=kl_instance_marg_pred_mean,
                              kl_marg_uniform=kl_marg_uniform.tolist(),
                              kl_marg_uniform_mean=kl_marg_uniform_mean)

            if phase != "samples":
                kl_marg_marg_data = td.kl_divergence(marginal_pred, marginals_data[phase])  # .mean().item()
                kl_marg_marg_data_mean = kl_marg_marg_data.mean().item()
                res[phase]["kl_marg_marg_data"] = kl_marg_marg_data.tolist()
                res[phase]["kl_marg_marg_data_mean"] = kl_marg_marg_data_mean

        print("Saving KNN stats!")
        pickle.dump(res, open(knn_preds_stats_file, "wb"))


def gather_avg_data_sample_per_class():
    if not os.path.isfile(AVG_DATA_SAMPLE_FILE):
        val_data_X, val_data_y = get_n_data_samples_x_y(image_dataset_name="bmnist", N_samples=9000, phase="valid")

        avg_digits_X_flat = []

        for i in range(10):
            val_data_X_i = val_data_X[val_data_y == i].reshape(-1, 28 * 28).mean(dim=0)
            avg_digits_X_flat.append(val_data_X_i)

        # fig, axs = plt.subplots(ncols=5, nrows=2)
        #
        # for i in range(10):
        #     row = i // 5
        #     col = i % 5
        #
        #     im = axs[row, col].imshow(avg_digits_X_flat[i].reshape(28, 28), cmap="Greys")
        #
        # plt.colorbar(im)
        # plt.show()

        pickle.dump(avg_digits_X_flat, open(AVG_DATA_SAMPLE_FILE, "wb"))
    else:
        avg_digits_X_flat = pickle.load(open(AVG_DATA_SAMPLE_FILE, "rb"))

    avg_digits_X_flat = torch.stack(avg_digits_X_flat, dim=0)
    assert avg_digits_X_flat.shape[0] == 10, "gather_avg_data_sample_per_class: stacking went wrong, invalid shape"
    return avg_digits_X_flat

def data_space_collect_stats(x, preds, val_data_y, avg_digits_X_flat):
    uniform = td.Categorical(probs=torch.FloatTensor([0.1 for _ in range(10)]))

    res = dict()

    fracs = []
    data_fracs = []
    L2_all_classes = []
    L0_all_classes = []
    L2_all_data = []

    data_x_i_avgs = []
    x_i_avgs = []

    L0_all = torch.flatten(x, start_dim=1).sum(dim=-1)

    for i in range(10):
        x_i = torch.flatten(x[preds == i], start_dim=1)

        # val_data_x_i = torch.flatten(val_data_X[val_data_y == i], start_dim=1)

        val_data_y_i = val_data_y[val_data_y == i]

        pred_frac_i = len(x_i) / len(x)
        val_data_frac_i = len(val_data_y_i) / len(val_data_y)

        # val_data_x_i_avg = torch.mean(val_data_x_i, dim=0)

        val_data_x_i_avg = avg_digits_X_flat[i, :]

        x_i_avg = torch.mean(x_i, dim=0)

        assert val_data_x_i_avg.shape == x_i_avg.shape, "avg data sample and avg sample must have same size " \
                                                        "to compare "

        data_x_i_avgs.append(val_data_x_i_avg)
        x_i_avgs.append(x_i_avg)

        L0_i = x_i.sum(-1).mean()
        L0_all_classes.append(L0_i.item())

        L2 = ((x_i - val_data_x_i_avg) ** 2).sum(dim=-1)
        L2_all_data.append(L2)
        L2_all_classes.append(L2.mean().item())

        fracs.append(pred_frac_i)
        data_fracs.append(val_data_frac_i)

    L2_all_data = torch.cat(L2_all_data, dim=0)

    sample_dist = td.Categorical(probs=torch.FloatTensor(fracs))
    data_dist = td.Categorical(probs=torch.FloatTensor(data_fracs))

    kl_div_sample_dist_from_uniform = td.kl_divergence(sample_dist, uniform).item()
    kl_div_sample_dist_from_data_dist = td.kl_divergence(sample_dist, data_dist).item()

    res["L0_avg"] = L0_all.mean().item()
    res["L0_all"] = L0_all.tolist()
    res["L0_avg_per_class"] = L0_all_classes

    res["L2_avg"] = np.mean(L2_all_classes)
    res["L2_avg_per_class"] = L2_all_classes
    res["L2_all"] = L2_all_data.tolist()

    res["KL_marg_sample_dist_data_dist"] = kl_div_sample_dist_from_data_dist
    res["KL_marg_sample_dist_uniform"] = kl_div_sample_dist_from_uniform
    res["marg_sample_dist"] = sample_dist

    res["class_fracs"] = fracs
    res["data_class_fracs"] = data_fracs
    res["data_x_i_avgs"] = data_x_i_avgs
    res["x_i_avgs"] = x_i_avgs

    return res

def data_space_stats():
    val_data_X, val_data_y = get_n_data_samples_x_y(image_dataset_name="bmnist", N_samples=1000, phase="valid")

    avg_digits_X_flat = gather_avg_data_sample_per_class()
    # ------------------------------------------------------------
    # Collect all the stats for the data group
    save_dir = f"{ANALYSIS_DIR}/data_group"
    os.makedirs(save_dir, exist_ok=True)
    save_file = f"{save_dir}/{DATA_SPACE_STATS}"

    if not os.path.isfile(save_file):
        print("Collecting dataspace stats for the data samples, to form a data group in analysis!")

        all_res = dict()

        # Collect the stats for the data itself
        for phase in ["train", "valid", "test"]:
            print(phase)

            x, preds = get_n_data_samples_x_y(image_dataset_name="bmnist", N_samples=1000, phase=phase)
            res = data_space_collect_stats(x, preds, val_data_y, avg_digits_X_flat)

            all_res[phase] = res

        all_res["samples"] = all_res["test"]

        print("Saving data space stats!")
        pickle.dump(all_res, open(save_file, "wb"))

    # ------------------------------------------------------------
    # Collect the stats for all the runs
    for i, run_name in enumerate(os.listdir(ANALYSIS_DIR)):
        # Skip all that is not a directory
        if os.path.isfile(f"{ANALYSIS_DIR}/{run_name}"):
            continue

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        save_file = f"{save_dir}/{DATA_SPACE_STATS}"

        # if os.path.isfile(save_file):
        #     print("Did this one already, skipping!")

        if not (os.path.isfile(f"{save_dir}/{KNN_PREDICT_RECONSTRUCTIONS_FILE}") or
                os.path.isfile(f"{save_dir}/{KNN_PREDICT_SAMPLES_FILE}")):
            print("No KNN predictions yet. Run knn_predictions_for_samples_reconstructions")
            continue

        knn_preds_samples = pickle.load(open(f"{save_dir}/{KNN_PREDICT_SAMPLES_FILE}", "rb"))
        knn_preds_recons = pickle.load(open(f"{save_dir}/{KNN_PREDICT_RECONSTRUCTIONS_FILE}", "rb"))

        encode_reconstruct = torch.load(f"{save_dir}/{ENCODE_RECONSTUCT_FILE}")
        samples = torch.load(f"{save_dir}/{SAMPLE_FILE}")

        data = {'samples': samples, **encode_reconstruct}
        knn_preds = {"samples": knn_preds_samples, **knn_preds_recons}

        all_results = dict()

        for phase in ["train", "valid", "test", "samples"]:

            if phase != "samples":
                x = data[phase]["reconstruct"]
            else:
                x = data[phase]['x']

            preds = knn_preds[phase]["preds"]
            assert len(preds) == len(x), "the predictions and samples are expected to be of the same length"

            res = data_space_collect_stats(x, preds, val_data_y, avg_digits_X_flat)
            all_results[phase] = res

        print("Saving data space stats!")
        pickle.dump(all_results, open(save_file, "wb"))
