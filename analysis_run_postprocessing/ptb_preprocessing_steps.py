

import os
import pathlib

if "cbarkhof" in str(pathlib.Path(__file__).parent.resolve()):
    CODE_DIR = "/home/cbarkhof/fall-2021"
else:
    CODE_DIR = "/"

import sys
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.utils.data import TensorDataset, DataLoader
from vae_model.distributions import AutoRegressiveDistribution

from torch_two_sample import MMDStatistic

import numpy as np
import pickle
from train import Trainer
from utils import load_checkpoint_model_for_eval
from analysis.MNIST_analysis.analysis_utils import get_wandb_runs, get_test_validation_loader, get_n_data_samples_x_y

import pandas as pd

pd.options.display.float_format = '{:.6f}'.format

import seaborn as sns;

sns.set()

ANALYSIS_DIR = f"{CODE_DIR}/analysis/analysis-files"
CHECKPOINT_DIR = f"{CODE_DIR}/run_files/checkpoints"

# ENCODE_RECONSTUCT_FILE = f"encode-reconstruct-test-valid.pt"
SAMPLE_FILE = f"generative-samples.pt"
CONDITIONAL_SAMPLE_FILE = f"generative-conditional-samples.pt"

SURPRISAL_SAMPLE_FILE = "surprisal_sample.pt"
SURPRISAL_DATA_FILE = "surprisal_data.pt"
SURPRISAL_COND_SAMPLE_FILE = "surprisal_conditional_sample.pt"

TEST_VALID_EVAL_FILE = "test-valid-results.pt"

BASE_ALPHA = 0.001
MMD_REDO_FILE = f"mmd_redo-base-alpha-{BASE_ALPHA}.pt"

# -------------------------------------------------------------------------------------------------------------------
# STEP 0: Fetch run data from W&B based on and filter on run_name prefixes
def make_run_overview_df(prefixes, add_data_group=False):
    runs = get_wandb_runs(entity="fall-2021-vae-claartje-wilker", project="fall-2021-VAE")

    q_z_x_mapper = {
        "conditional_gaussian_made": "MADE",
        "independent_gaussian": "ind. Gauss."
    }

    decoder_mapper = {
        "strong_distil_roberta_decoder": "Strong roBERTa",
        "weak_distil_roberta_decoder": "Weak roBERTa",
        "weak_memory_distil_roberta_decoder": "Weak-M roBERTa",
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
                dataset = run.config["language_dataset_name"]

                dropout = run.config["strong_roberta_decoder_embedding_dropout"]
                if dropout:
                    p = run.config["strong_roberta_decoder_embedding_dropout_prob"]
                    dropout_str = f" | drop-{p} |"
                else:
                    p = 0.0
                    dropout_str = ""

                l_rate, l_mmd, beta, mdr_val, fb = 0, 0, 0, 0, 0

                dec = run.config["decoder_network_type"]

                if obj == "INFO-VAE":
                    l_rate = run.config["info_lambda_1_rate"]
                    l_mmd = run.config["info_lambda_2_mmd"]
                    clean_name = f"{dataset}{dropout_str} INFO-VAE l_Rate {l_rate} l_MMD {l_mmd} dec: {decoder_mapper[dec]}"
                elif obj == "BETA-VAE":
                    beta = run.config["beta_beta"]
                    clean_name = f"{dataset}{dropout_str} BETA-VAE beta {beta} dec: {decoder_mapper[dec]}"
                elif obj == "MDR-VAE":
                    mdr_val = run.config["mdr_value"]
                    clean_name = f"{dataset}{dropout_str} MDR-VAE {mdr_val} dec: {decoder_mapper[dec]}"
                elif obj == "FB-VAE":
                    fb = run.config["free_bits"]
                    clean_name = f"{dataset}{dropout_str} FB-VAE {fb} dec: {decoder_mapper[dec]}"
                else:
                    print(f"make_run_overview_df: No clean name builder for this objective: {obj}")
                    continue

                d = {
                    "objective": obj,
                    "dataset": dataset,
                    "l_rate": l_rate,
                    "dropout": p,
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

    missing = {'dir': True, 'surprisal_data': True, 'surprisal_samples': True, 'mmd_redo': True,
               'surprisal_cond_samples': True, 'samples': True, 'cond_samples': True,
               'test_valid_eval': True}

    for run_name in df["run_name"].values:
        save_dir = f"{ANALYSIS_DIR}/{run_name}"

        # skip things that are not directories
        if os.path.isfile(save_dir):
            continue

        missing_run = missing.copy()

        if os.path.isdir(save_dir):
            missing_run["dir"] = False

        # (CONDITIONAL) SAMPLES
        if os.path.isfile(f"{save_dir}/{SAMPLE_FILE}"):
            missing_run["samples"] = False
        if os.path.isfile(f"{save_dir}/{CONDITIONAL_SAMPLE_FILE}"):
            missing_run["cond_samples"] = False

        # TEST / VALID SIMPLE EVAL
        if os.path.isfile(f"{save_dir}/{TEST_VALID_EVAL_FILE}"):
            missing_run["test_valid_eval"] = False

        # MMD REDO
        if os.path.isfile(f"{save_dir}/{MMD_REDO_FILE}"):
            missing_run["mmd_redo"] = False

        # SURPRISAL
        if os.path.isfile(f"{save_dir}/{SURPRISAL_DATA_FILE}"):
            missing_run["surprisal_data"] = False
        if os.path.isfile(f"{save_dir}/{SURPRISAL_SAMPLE_FILE}"):
            missing_run["surprisal_samples"] = False
        if os.path.isfile(f"{save_dir}/{SURPRISAL_COND_SAMPLE_FILE}"):
            missing_run["surprisal_cond_samples"] = False

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
def conditional_sample(data_loaders, vae_model, tok, device, conditional_sample_file, n_sample_batches=10, include_train=True):
    phases = ["train", "valid", "test"] if include_train else ["test", "valid"]
    conditional_sample_d = {phase: dict(z=[], mean=[], scale=[], cond_sample_x=[], condtional_sample_text=[],
                                        original_input_text=[], original_input_ids=[],
                                        original_attention_mask=[]) for phase in phases}

    for phase, data_loader in data_loaders.items():
        for i, x_in in enumerate(data_loader):
            print(f"{phase} {i}/{len(data_loader)}", end="\r")

            x_in = (x_in["input_ids"].to(device), x_in["attention_mask"].to(device))

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
            conditional_sample_d[phase]["z"].append(z_post.squeeze(0))
            conditional_sample_d[phase]["mean"].append(mean)
            conditional_sample_d[phase]["scale"].append(scale)

            # Conditional sampling
            # [Sx, Sz, B, L] -> [B, L]
            conditional_sample_x = vae_model.gen_model.sample_generative_model(z=z_post,
                                                                               Sx=1, Sz=1, return_z=False,
                                                                               device=device).squeeze(0).squeeze(0)

            conditional_sample_d[phase]["cond_sample_x"].append(conditional_sample_x)
            sample_text = convert_tensor_preds_to_text_list(tok, conditional_sample_x)
            conditional_sample_d[phase]["condtional_sample_text"].extend(sample_text)
            conditional_sample_d[phase]["original_input_ids"].append(x_in[0])
            original_text = convert_tensor_preds_to_text_list(tok, x_in[0])
            conditional_sample_d[phase]["original_input_text"].extend(original_text)
            conditional_sample_d[phase]["original_attention_mask"].append(x_in[1])

            if i + 1 == n_sample_batches:
                break

    # Cat
    for phase in phases:
        for key in ["z", "mean", "scale", "cond_sample_x", "original_input_ids", "original_attention_mask"]:
            conditional_sample_d[phase][key] = torch.cat(conditional_sample_d[phase][key], dim=0).cpu()
            print(phase, key, conditional_sample_d[phase][key].shape)

    # Calc MMD
    for phase in phases:
        conditional_sample_d[phase]["MMD"] = mmd_fn(z_post=conditional_sample_d[phase]["z"])

    print("Saving conditonal samplings!")
    torch.save(conditional_sample_d, conditional_sample_file)


def convert_tensor_preds_to_text_list(tok, sample_x):
    # Check where the eos_token is in the predictions, if not there set to max_len
    lens = [a.index(tok.eos_token_id) if tok.eos_token_id in a else len(a) for a in sample_x.tolist()]
    text_samples = []
    for j in range(len(lens)):
        text_samples.append(tok.decode(sample_x[j, :lens[j]]))
    return text_samples


def sample(vae_model, tok, n_sample_batches, device, sample_batch_size, sample_file):
    samples = dict(z=[], x=[], text=[])

    for i in range(n_sample_batches):
        print(f"sample batch {i}/{n_sample_batches}", end="\r")

        sample_z, sample_x = vae_model.gen_model.sample_generative_model(Sx=1, Sz=sample_batch_size,
                                                                         return_z=True, device=device)
        # sample x = [Sx, Sz, B, L]
        sample_x = sample_x.squeeze(0).squeeze(1)
        # sample z = [Sz, B, D]
        sample_z = sample_z.squeeze(1)

        samples["z"].append(sample_z)
        samples["x"].append(sample_x)
        samples["text"].extend(convert_tensor_preds_to_text_list(tok, sample_x))

    for key in ["z", "x"]:
        samples[key] = torch.cat(samples[key], dim=0).cpu()
        print(key, samples[key].shape)

    print("Saving samples!")
    torch.save(samples, sample_file)


def collect_samples(df, tok, device="cuda:0", include_train=True, n_sample_batches=20, sample_batch_size=100, reverse=False):
    assert "run_name" in df.columns, "collect_samples: the DF must have a run_name column"
    assert "dataset" in df.columns, "collect_samples: the DF must have a dataset column"

    all_data_loaders = dict()
    for dataset in df.dataset.unique():
        print(f"Loading {dataset}")
        all_data_loaders[dataset] = get_test_validation_loader(language_dataset_name=dataset,
                                                               image_or_language="language",
                                                               batch_size=sample_batch_size,
                                                               num_workers=3, include_train=include_train)

    run_names = list(df["run_name"].values)
    datasets = list(df["dataset"].values)

    if reverse:
        run_names.reverse()
        datasets.reverse()

    for i, (run_name, dataset) in enumerate(zip(run_names, datasets)):
        print(i, run_name, dataset)

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        conditional_sample_file = f"{save_dir}/{CONDITIONAL_SAMPLE_FILE}"
        sample_file = f"{save_dir}/{SAMPLE_FILE}"

        done_sample, done_cond_sample = False, False

        if os.path.isfile(sample_file):
            done_sample = True

        if os.path.isfile(conditional_sample_file):
            done_cond_sample = True

        if done_sample and done_cond_sample:
            print("Done this run already, continue")
            continue

        checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
        vae_model = load_checkpoint_model_for_eval(checkpoint_path, map_location="cuda:0", return_args=False).to(device)

        # SAMPLE
        if not done_sample:
            with torch.no_grad():
                sample(vae_model=vae_model, tok=tok, n_sample_batches=n_sample_batches, device=device,
                       sample_batch_size=sample_batch_size, sample_file=sample_file)
        else:
            print("Done sampling already, skipping this step.")


        # CONDITIONAL SAMPLE
        if not done_cond_sample:
            print(i, f"conditional sampling start: {n_sample_batches} x {sample_batch_size} = {n_sample_batches*sample_batch_size} samples")
            with torch.no_grad():
                conditional_sample(data_loaders=all_data_loaders[dataset], vae_model=vae_model, tok=tok, device=device,
                                   n_sample_batches=n_sample_batches, conditional_sample_file=conditional_sample_file,
                                   include_train=include_train)
        else:
            print("Done conditional sampling already, skipping this step.")


# -------------------------------------------------------------------------------------------------------------------
# STEP 2: Gather surprisal stats for data, reconstructions and model samples
def collect_data_surprisal_stats(df, batch_size_surprisal=4, iw_ll_n_samples=40, num_workers=3,
                                 max_data_samples=2000, device="cuda:0"):

    all_data_loaders = dict()
    for dataset in df.dataset.unique():
        print(f"Loading {dataset}")
        all_data_loaders[dataset] = get_test_validation_loader(language_dataset_name=dataset,
                                                               image_or_language="language",
                                                               batch_size=batch_size_surprisal,
                                                               num_workers=num_workers, include_train=True)

    for row_idx, row in df.iterrows():
        print(row_idx, row.run_name)

        suprisal_data_file = f"{ANALYSIS_DIR}/{row.run_name}/{SURPRISAL_DATA_FILE}"

        if os.path.isfile(suprisal_data_file):
            print("Done this already, skipping it!")
            continue

        checkpoint_path = f"{CHECKPOINT_DIR}/{row.run_name}.pt"
        vae_model = load_checkpoint_model_for_eval(checkpoint_path, map_location=device, return_args=False).to(device)

        iw_ll_dict = dict()
        for phase in ["train", "valid", "test"]:
            loader = all_data_loaders[row.dataset][phase]

            iw_lls = []

            for i, batch in enumerate(loader):
                if i * batch_size_surprisal > max_data_samples:
                    break

                print(f"{phase} - {i + 1:3d}/{len(loader)}", end='\r')

                input_ids, attention_mask = batch["input_ids"].to(device).long(), batch["attention_mask"].to(device).float()
                x_in = (input_ids, attention_mask)

                with torch.no_grad():
                    iw_ll = vae_model.estimate_log_likelihood_batch(x_in, n_samples=iw_ll_n_samples,
                                                                    image_or_language="language",
                                                                    decoder_network_type=row.decoder,
                                                                    per_bit=False)
                    iw_lls.append(iw_ll)

            iw_lls = torch.cat(iw_lls, dim=0)

            # done with phase
            iw_ll_dict[phase] = dict(
                    iw_lls=iw_lls.cpu(),
                    iw_ll_mean=iw_lls.mean().item(),
                    iw_ll_std=iw_lls.std().item()
                )

        # done with all phases
        torch.save(iw_ll_dict, suprisal_data_file)


def convert_sample_ids_to_new_inputs(samples, weak, bos_token_id=0, eos_token_id=2, pad_token_id=1):
    input_ids = []
    attention_masks = []

    for i in range(samples.shape[0]):
        if weak:
            s_i = samples[i, :]
            s_i[0] = bos_token_id
        else:
            s_i = torch.cat([torch.Tensor([bos_token_id]).long(), samples[i, :]])

        if weak:
            s_i[0] = bos_token_id

        length = s_i.tolist().index(eos_token_id) + 1 if eos_token_id in s_i else len(s_i)
        mask = torch.cumsum(torch.ones_like(s_i), dim=-1) > length

        s_i[mask] = pad_token_id
        s_i[length - 1] = eos_token_id

        input_ids.append(s_i.long().unsqueeze(0))
        attention_masks.append((~mask).float().unsqueeze(0))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def collect_sample_surprisal_stats(overview_df, batch_size_surprisal=4, iw_ll_n_samples=40, device="cuda:0"):

    for row_idx, row in overview_df.iterrows():
        print(row_idx, row.run_name)

        conditional_sample_file = f"{ANALYSIS_DIR}/{row.run_name}/{CONDITIONAL_SAMPLE_FILE}"
        suprisal_conditional_sample_file = f"{ANALYSIS_DIR}/{row.run_name}/{SURPRISAL_COND_SAMPLE_FILE}"

        sample_file = f"{ANALYSIS_DIR}/{row.run_name}/{SAMPLE_FILE}"
        surprisal_sample_file = f"{ANALYSIS_DIR}/{row.run_name}/{SURPRISAL_SAMPLE_FILE}"

        # Weak decoder's samples start with <s> tokens, strong decoders not
        weak = True if "weak" in row.decoder else False

        checkpoint_path = f"{CHECKPOINT_DIR}/{row.run_name}.pt"
        vae_model = load_checkpoint_model_for_eval(checkpoint_path, map_location=device, return_args=False).to(device)

        # Samples conditional generation
        if os.path.exists(conditional_sample_file) and not os.path.exists(suprisal_conditional_sample_file):
            con_samples = torch.load(conditional_sample_file)

            iw_ll_dict = dict()
            for phase in ["train", "valid", "test"]:

                input_ids, attention_masks = convert_sample_ids_to_new_inputs(
                    samples=con_samples[phase]["cond_sample_x"],
                    weak=weak,
                    bos_token_id=0,
                    eos_token_id=2,
                    pad_token_id=0)

                print(phase, input_ids.shape, attention_masks.shape)

                sample_loader = DataLoader(TensorDataset(input_ids, attention_masks),
                                           batch_size=batch_size_surprisal)

                iw_lls = []

                for i, (input_ids, attention_mask) in enumerate(sample_loader):
                    print(f"{i + 1:3d}/{len(sample_loader)}")
                    input_ids, attention_mask = input_ids.to(device).long(), attention_mask.to(device).float()
                    x_in = (input_ids, attention_mask)

                    with torch.no_grad():
                        iw_ll = vae_model.estimate_log_likelihood_batch(x_in, n_samples=iw_ll_n_samples,
                                                                        image_or_language="language",
                                                                        decoder_network_type=row.decoder,
                                                                        per_bit=False)
                        iw_lls.append(iw_ll)

                iw_lls = torch.cat(iw_lls)

                iw_ll_dict[phase] = dict(
                    iw_lls=iw_lls.cpu(),
                    iw_ll_mean=iw_lls.mean().item(),
                    iw_ll_std=iw_lls.std().item()
                )

            torch.save(iw_ll_dict, suprisal_conditional_sample_file)

        # Generative samples surprisal
        if os.path.exists(sample_file) and not os.path.exists(surprisal_sample_file):

            samples = torch.load(sample_file)

            input_ids, attention_masks = convert_sample_ids_to_new_inputs(samples=samples["x"],
                                                                          weak=weak,
                                                                          bos_token_id=0,
                                                                          eos_token_id=2,
                                                                          pad_token_id=0)
            print(input_ids.shape, attention_masks.shape)

            sample_loader = DataLoader(TensorDataset(input_ids, attention_masks),
                                       batch_size=batch_size_surprisal)

            iw_lls = []

            for i, (input_ids, attention_mask) in enumerate(sample_loader):
                print(f"{i + 1:3d}/{len(sample_loader)}")
                input_ids, attention_mask = input_ids.to(device).long(), attention_mask.to(device).float()
                x_in = (input_ids, attention_mask)

                with torch.no_grad():
                    iw_ll = vae_model.estimate_log_likelihood_batch(x_in, n_samples=iw_ll_n_samples,
                                                                    image_or_language="language",
                                                                    decoder_network_type=row.decoder,
                                                                    per_bit=False)
                    iw_lls.append(iw_ll)

            iw_lls = torch.cat(iw_lls)

            iw_ll_dict = dict(
                iw_lls=iw_lls.cpu(),
                iw_ll_mean=iw_lls.mean().item(),
                iw_ll_std=iw_lls.std().item()
            )

            torch.save(iw_ll_dict, surprisal_sample_file)

# -------------------------------------------------------------------------------------------------------------------
# STEP 3: Simple evaluate valid and test set (global statistics)
def simple_evaluate_valid_test(df, device="cuda:0", batch_size=100, num_workers=3):
    assert "run_name" in df.columns, "simple_evaluate_valid_test: the DF must have a run_name column"
    save_metrics = ['elbo', 'distortion', 'kl_prior_post', 'mean_mean', 'std_across_x_mean',
                    'std_across_z_mean', 'mean_scale', 'std_across_x_scale', 'std_across_z_scale'] #'mmd',

    print(f"First re-computing MMD with BASE_ALPHA = {BASE_ALPHA}")
    redo_calc_mmd(df, base_alpha=BASE_ALPHA)

    all_data_loaders = dict()
    for dataset in df.dataset.unique():
        print(f"Loading {dataset}")
        all_data_loaders[dataset] = get_test_validation_loader(language_dataset_name=dataset,
                                                               image_or_language="language",
                                                               batch_size=batch_size,
                                                               num_workers=num_workers, include_train=False)

    run_names = list(df["run_name"].values)
    datasets = list(df["dataset"].values)

    for i, (run_name, dataset) in enumerate(zip(run_names, datasets)):
        print(i, run_name)

        save_dir = f"{ANALYSIS_DIR}/{run_name}"
        os.makedirs(save_dir, exist_ok=True)

        # RESULT FILE
        result_file = f"{save_dir}/{TEST_VALID_EVAL_FILE}"
        conditional_sample_file = f"{save_dir}/{CONDITIONAL_SAMPLE_FILE}"

        results = dict()

        if os.path.isfile(result_file):
            print("Did this one already, skipping it for now.")
            continue

        if not os.path.isfile(conditional_sample_file):
            print("No conditional samples to retrieve MMD from, skipping it for now. "
                  "Run collect samples for this run first!")
            continue

        conditional_samples = torch.load(conditional_sample_file)

        with torch.no_grad():

            # CHECKPOINT
            checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
            vae_model, args = load_checkpoint_model_for_eval(checkpoint_path, map_location=device, return_args=True)
            vae_model = vae_model.to(device)
            trainer = Trainer(args=args, dataset=None, vae_model=vae_model, device=device)

            # TEST / VALIDATE
            for phase, loader in all_data_loaders[dataset].items():

                results[phase] = dict()
                res = trainer.test(device=device, loader=loader)

                for k, v in res.items():
                    if k in save_metrics:
                        results[phase][k] = v
                        results[phase][k + " mean"] = np.mean(v)

                print(conditional_samples[phase]["MMD"])
                results[phase]["MMD"] = conditional_samples[phase]["MMD"]

            torch.save(results, result_file)


def mmd_fn(z_post, alphas=[0.001, 0.002, 0.003], max_samples=1000):
    # Random row shuffle
    z_post = z_post[torch.randperm(z_post.size()[0])]

    # Random subset
    z_post = z_post[:max_samples, :]
    with torch.no_grad():
        # [S, B, D] -> [B, D]
        prior_sample = torch.randn_like(z_post)  # .to(z_post.device)

        n_1, n_2 = len(z_post), len(prior_sample)
        MMD_stat = MMDStatistic(n_1, n_2)

        tts_mmd = MMD_stat(z_post, prior_sample, alphas, ret_matrix=False)

        if torch.is_tensor(tts_mmd):
            tts_mmd = tts_mmd.item()

        return tts_mmd


def redo_calc_mmd(df, base_alpha=0.001):
    alphas = [base_alpha * i for i in range(1, 5)]

    for row_index, row in df.iterrows():
        print(row_index)

        conditional_sample_file = f"{ANALYSIS_DIR}/{row.run_name}/{CONDITIONAL_SAMPLE_FILE}"
        if not os.path.isfile(conditional_sample_file):
            print("No conditional samples for this model yet, please re-run collect_samples.")
            continue

        mmd_save_file = f"{ANALYSIS_DIR}/{row.run_name}/{MMD_REDO_FILE}"

        if not os.path.exists(mmd_save_file):
            conditional_samples = torch.load(conditional_sample_file)
            mmd_dict = dict()

            for phase in ["train", "valid", "test"]:
                mmd = mmd_fn(conditional_samples[phase]["z"], alphas=alphas, max_samples=2000)
                mmd_dict[phase] = mmd

            torch.save(mmd_dict, mmd_save_file)