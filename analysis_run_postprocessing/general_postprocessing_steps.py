import os
import sys
import pathlib

if "cbarkhof" in str(pathlib.Path(__file__).parent.resolve()):
    CODE_DIR = "/home/cbarkhof/fall-2021"
else:
    CODE_DIR = "/"

sys.path.append(CODE_DIR)

EXPORT_DIR = f"{CODE_DIR}/analysis_run_postprocessing/run_postprocess_files"
CHECKPOINT_DIR = f"{CODE_DIR}/run_files/checkpoints"

SAMPLE_FILE = f"generative-samples.pt"
CONDITIONAL_SAMPLE_FILE = f"generative-conditional-samples.pt"
ENCODING_FILE = "encodings.pt"

EVALUATION_RESULT_FILE = "evaluation-results.pt"
SURPRISAL_DATA_FILE = "surprisal_data.pt"

import utils
import train
from dataset_dataloader import get_test_validation_loader
from torch_two_sample import MMDStatistic

from vae_model.distributions import AutoRegressiveDistribution

import torch
import torch.distributions as td
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Make an overview of runs you want to process based on W&B logs
def make_run_overview_df(prefixes):
    runs = utils.get_wandb_runs(entity="fall-2021-vae-claartje-wilker", project="fall-2021-VAE")

    decoder_mapper = {
        "conditional_made_decoder": "MADE",
        "cond_pixel_cnn_pp": "PixelCNN++",
        "basic_deconv_decoder": "CNN.T",
        "basic_mlp_decoder": "MLP",
        "strong_distil_roberta_decoder": "Strong roBERTa",
        "weak_distil_roberta_decoder": "Weak roBERTa",
        "weak_memory_distil_roberta_decoder": "Weak-M roBERTa"
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

                if run.config["image_or_language"] == "image":
                    dataset = run.config["image_dataset_name"]
                else:
                    dataset = run.config["language_dataset_name"]

                d = {
                    "objective": obj,
                    "l_rate": l_rate,
                    "dataset": dataset,
                    "image_or_language": run.config["image_or_language"],
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


def overview_of_missing_analysis(df):
    all_missing = dict()

    missing = {'dir': True, 'surprisal_data': True, 'encode': True, 'samples': True, 'cond_samples': True,
               'evaluation': True}

    SAMPLE_FILE = f"generative-samples.pt"
    CONDITIONAL_SAMPLE_FILE = f"generative-conditional-samples.pt"
    ENCODING_FILE = "encodings.pt"

    EVALUATION_RESULT_FILE = "evaluation-results.pt"
    SURPRISAL_DATA_FILE = "surprisal_data.pt"

    for run_name in df["run_name"].values:
        save_dir = f"{EXPORT_DIR}/{run_name}"

        # skip things that are not directories
        if os.path.isfile(save_dir):
            continue

        missing_run = missing.copy()

        if os.path.isdir(save_dir):
            missing_run["dir"] = False

        # ENCODE / SAMPLES
        if os.path.isfile(f"{save_dir}/{ENCODING_FILE}"):
            missing_run["encode"] = False
        if os.path.isfile(f"{save_dir}/{SAMPLE_FILE}"):
            missing_run["samples"] = False
        if os.path.isfile(f"{save_dir}/{CONDITIONAL_SAMPLE_FILE}"):
            missing_run["cond_samples"] = False

        # TEST / VALID SIMPLE EVAL
        if os.path.isfile(f"{save_dir}/{EVALUATION_RESULT_FILE}"):
            missing_run["evaluation"] = False

        # SURPRISAL
        if os.path.isfile(f"{save_dir}/{SURPRISAL_DATA_FILE}"):
            missing_run["surprisal_data"] = False

        # any_missing = False
        # for k, v in missing_run.items():
        #     if v is True:
        #         any_missing = True
        #-
        #     if any_missing:
        #         break
        #
        # if any_missing:
        all_missing[run_name] = missing_run

    df_missing = pd.DataFrame(all_missing).transpose()

    return df_missing


# -----------------------------------------------------------------------------
# Encodings (z, mean, scales)
def encode(data_loaders, model, device, encode_file, include_train=False, image_language="image"):
    phases = ["train", "valid", "test"] if include_train else ["test", "valid"]
    encodings = {phase: dict(z=[], mean=[], scale=[]) for phase in phases}

    for phase, data_loader in data_loaders.items():
        for i, batch in enumerate(data_loader):
            print(f"{phase} {i}/{len(data_loader)}", end="\r")

            if image_language == "image":
                x_in, y = batch
                x_in = x_in.to(device)
            else:
                x_in = (batch["input_ids"].to(device), batch["attention_mask"].to(device))

            # ENCODE

            # Infer the approximate posterior q(z|x) and sample from it to obtain z_post
            # [B, D], [S, B, D]
            q_z_x, z_post = model.inf_model(x_in=x_in, n_samples=1)

            if isinstance(q_z_x, td.Independent):
                mean, scale = q_z_x.base_dist.loc, q_z_x.base_dist.scale

            elif isinstance(q_z_x, AutoRegressiveDistribution):
                (mean, scale) = q_z_x.params

            # Normal normal
            else:
                mean, scale = q_z_x.loc, q_z_x.scale

            # mean, scale: B, D
            # z_post: S, B, D = 1, B, D

            encodings[phase]["z"].append(z_post.squeeze(0))
            encodings[phase]["mean"].append(mean)
            encodings[phase]["scale"].append(scale)

    # Cat
    for phase in phases:
        for key in ["z", "mean", "scale"]:
            encodings[phase][key] = torch.cat(encodings[phase][key], dim=0).cpu()

    print("Saving encodings!")
    torch.save(encodings, encode_file)


# -----------------------------------------------------------------------------
# Samples (conditionally & unconditionally)
def image_sample(vae_model, n_sample_batches, device, sample_batch_size, sample_file):
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


def image_conditional_sample(data_loaders, vae_model, device, conditional_sample_file, n_sample_batches=10,
                             include_train=True):
    phases = ["train", "valid", "test"] if include_train else ["test", "valid"]
    conditional_sample_d = {phase: dict(z=[], mean=[], scale=[], cond_sample_x=[],
                                        original_x=[], original_y=[]) for phase in phases}

    for phase, data_loader in data_loaders.items():

        if phase != "valid":
            print("only perform conditional sampling based on validation samples for now.")
            continue

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
            conditional_sample_d[phase]["z"].append(z_post.squeeze(0))
            conditional_sample_d[phase]["mean"].append(mean)
            conditional_sample_d[phase]["scale"].append(scale)

            # Conditional sampling
            # [Sx, Sz, B, C, W, H] -> [B, C, W, H]
            conditional_sample_x = vae_model.gen_model.sample_generative_model(z=z_post,
                                                                               Sx=1, Sz=1, return_z=False,
                                                                               device=device).squeeze(0).squeeze(
                0)
            conditional_sample_d[phase]["cond_sample_x"].append(conditional_sample_x)
            conditional_sample_d[phase]["original_x"].append(x_in)
            conditional_sample_d[phase]["original_y"].append(y)

            if i + 1 == n_sample_batches:
                break

    # Cat
    # for phase in conditional_sample_d.keys():
    for phase in ["valid"]:
        print("Warning! altered code, only looping over valid in image_conditional_sample.")
        for key in ["z", "mean", "scale", "cond_sample_x", "original_x", "original_y"]:
            conditional_sample_d[phase][key] = torch.cat(conditional_sample_d[phase][key], dim=0).cpu()

    print("Saving conditonal samplings!")
    torch.save(conditional_sample_d, conditional_sample_file)


def language_conditional_sample(data_loaders, vae_model, tok, device, conditional_sample_file, n_sample_batches=10,
                                include_train=True):
    phases = ["train", "valid", "test"] if include_train else ["test", "valid"]
    conditional_sample_d = {phase: dict(z=[], mean=[], scale=[], cond_sample_x=[], condtional_sample_text=[],
                                        original_input_text=[], original_input_ids=[],
                                        original_attention_mask=[]) for phase in phases}

    for phase, data_loader in data_loaders.items():

        if phase != "valid":
            print("only valid conditional sampling for now.")
            continue

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
    #for phase in conditional_sample_d.keys():
    for phase in ["valid"]:
        print("Warning! altered code, only looping over valid in language_conditional_sample.")
        for key in ["z", "mean", "scale", "cond_sample_x", "original_input_ids", "original_attention_mask"]:
            conditional_sample_d[phase][key] = torch.cat(conditional_sample_d[phase][key], dim=0).cpu()
            print(phase, key, conditional_sample_d[phase][key].shape)

    print("Saving conditonal samplings!")
    torch.save(conditional_sample_d, conditional_sample_file)


def convert_tensor_preds_to_text_list(tok, sample_x):
    # Check where the eos_token is in the predictions, if not there set to max_len
    lens = [a.index(tok.eos_token_id) if tok.eos_token_id in a else len(a) for a in sample_x.tolist()]
    text_samples = []
    for j in range(len(lens)):
        text_samples.append(tok.decode(sample_x[j, :lens[j]]))
    return text_samples


def language_sample(vae_model, tok, n_sample_batches, device, sample_batch_size, sample_file):
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


# -----------------------------------------------------------------------------
# Gather encodings and samples
def encode_sample_all(df, device="cuda:0", include_train=True, n_sample_batches=20, batch_size=128,
                      sample_batch_size=100, tok=None, force_recompute=False):
    assert "run_name" in df.columns, "encode_reconstruct_sample: the DF must have a run_name column"
    assert "dataset" in df.columns, "encode_reconstruct_sample: the DF must have a dataset column"
    assert "image_or_language" in df.columns, "encode_reconstruct_sample: the DF must have a image_or_language column"

    data_loaders = dict()

    for idx, row in df.iterrows():

        print(idx)

        # if "pixel" not in row.decoder:
        #     print("only doing pixel_cnn decoders for now.")
        #
        # if row.image_or_language == "image":
        #     print("focus on language now")
        #     continue

        if row.dataset not in data_loaders:
            data_loaders[row.dataset] = get_test_validation_loader(dataset_name=row.dataset,
                                                                   shuffle=False,
                                                                   batch_size=batch_size,
                                                                   num_workers=3,
                                                                   include_train=include_train)

        save_dir = f"{EXPORT_DIR}/{row.run_name}"
        os.makedirs(save_dir, exist_ok=True)

        encode_file = f"{save_dir}/{ENCODING_FILE}"
        conditional_sample_file = f"{save_dir}/{CONDITIONAL_SAMPLE_FILE}"
        sample_file = f"{save_dir}/{SAMPLE_FILE}"

        done_encode, done_sample, done_cond_sample = False, False, False

        if os.path.isfile(encode_file):
            done_encode = True

        if os.path.isfile(sample_file):
            done_sample = True

        if os.path.isfile(conditional_sample_file):
            done_cond_sample = True

        if done_encode and done_sample and done_cond_sample and not force_recompute:
            print("Done this run already, continue")
            continue

        checkpoint_path = f"{CHECKPOINT_DIR}/{row.run_name}.pt"
        model = utils.load_checkpoint_model_for_eval(checkpoint_path, map_location=device, return_args=False).to(
            device)

        # ENCODE
        if (not done_encode) or force_recompute:
            with torch.no_grad():
                encode(data_loaders=data_loaders[row.dataset], model=model, device=device,
                       encode_file=encode_file, include_train=include_train, image_language=row.image_or_language)
        else:
            print("Done encodings already, skipping.")

        # SAMPLE
        if (not done_sample) or force_recompute:
            with torch.no_grad():

                if row.dataset == "bmnist":
                    image_sample(vae_model=model, n_sample_batches=n_sample_batches, device=device,
                                 sample_batch_size=sample_batch_size, sample_file=sample_file)
                else:
                    language_sample(vae_model=model, tok=tok, n_sample_batches=n_sample_batches, device=device,
                                    sample_batch_size=sample_batch_size, sample_file=sample_file)
        else:
            print("Done sampling already, skipping.")

        # CONDITIONAL SAMPLE
        if (not done_cond_sample) or force_recompute:
            print(idx, f"conditional sampling start: {n_sample_batches} x {batch_size} = {n_sample_batches * batch_size} samples")
            with torch.no_grad():

                if row.dataset == "bmnist":
                    image_conditional_sample(data_loaders=data_loaders[row.dataset], vae_model=model, device=device,
                                             n_sample_batches=n_sample_batches, conditional_sample_file=conditional_sample_file,
                                             include_train=include_train)
                else:
                    language_conditional_sample(data_loaders=data_loaders[row.dataset], vae_model=model, tok=tok,
                                                device=device, conditional_sample_file=conditional_sample_file,
                                                n_sample_batches=n_sample_batches, include_train=include_train)
        else:
            print("Done conditional sampling already, skipping.")


# -----------------------------------------------------------------------------
# Surprisal computation
def surprisal_data(surprisal_data_file, data_loaders, vae_model, n_iw_samples, image_or_language, decoder_network_type, max_batches=20):
    print("Data")

    print("decoder_network_type", decoder_network_type)
    print("image_or_language", image_or_language)

    d = dict()
    for phase, data_loader in data_loaders.items():
        iw_lls, lens = vae_model.estimate_log_likelihood_dataset(data_loader, max_batches=max_batches,
                                                                 decoder_network_type=decoder_network_type,
                                                                 image_or_language=image_or_language,
                                                                 n_samples=n_iw_samples)

        if image_or_language == "language":
            # add two for <s> and </s>
            perplexity = np.exp(- np.array(iw_lls).mean() / (np.array(lens).mean()+2))
        else:
            perplexity = None

        d[phase] = dict(iw_lls=iw_lls, lens=lens, ppl=perplexity)

    print("Saving surprisal stats data")
    torch.save(d, surprisal_data_file)


def gather_surprisal_stats(run_df, device="cuda:0", include_train=True, force_recompute=False,
                           batch_size_surprisal=100, max_batches=20, n_iw_samples=50):
    data_loaders = dict()

    for idx, row in run_df.iterrows():

        if os.path.isfile(f"{EXPORT_DIR}/{row.run_name}"):
            continue

        if row.dataset not in data_loaders:
            data_loaders[row.dataset] = get_test_validation_loader(dataset_name=row.dataset,
                                                                   include_train=include_train,
                                                                   batch_size=batch_size_surprisal,
                                                                   num_workers=3)

        save_dir = f"{EXPORT_DIR}/{row.run_name}"
        os.makedirs(save_dir, exist_ok=True)

        surprisal_data_file = f"{save_dir}/{SURPRISAL_DATA_FILE}"

        if (not os.path.isfile(surprisal_data_file)) or force_recompute:
            checkpoint_path = f"{CHECKPOINT_DIR}/{row.run_name}.pt"
            vae_model, args = utils.load_checkpoint_model_for_eval(checkpoint_path, map_location=device,
                                                                   return_args=True)
            vae_model = vae_model.to(device)

            with torch.no_grad():
                surprisal_data(surprisal_data_file=surprisal_data_file, data_loaders=data_loaders[row.dataset],
                               decoder_network_type=row.decoder,
                               max_batches=max_batches, image_or_language=row.image_or_language,
                               vae_model=vae_model, n_iw_samples=n_iw_samples)
        else:
            print("did data already, skipping!")


# -----------------------------------------------------------------------------
# General evaluate
def mmd_fn(z_post, n_samples_mmd):
    # Random row shuffle
    z_post = z_post[torch.randperm(z_post.size()[0])]
    # Random subset
    z_post = z_post[:n_samples_mmd, :]
    with torch.no_grad():
        # [S, B, D] -> [B, D]
        prior_sample = torch.randn_like(z_post)

        alphas = [0.1 * i for i in range(5)]  # TODO: no clue for these...

        n_1, n_2 = len(z_post), len(prior_sample)
        MMD_stat = MMDStatistic(n_1, n_2)
        tts_mmd = MMD_stat(z_post, prior_sample, alphas, ret_matrix=False)
        return tts_mmd.item()


def evaluate(df, include_train=True, n_samples_mmd=5000, num_workers=3, batch_size=256, force_recompute=False, device="cuda:0"):
    assert "run_name" in df.columns, "simple_evaluate_valid_test: the DF must have a run_name column"
    assert "dataset" in df.columns, "simple_evaluate_valid_test: the DF must have a dataset column"

    save_metrics = ['elbo', 'distortion', 'kl_prior_post', 'mean_mean', 'std_across_x_mean',
                    'std_across_z_mean', 'mean_scale', 'std_across_x_scale', 'std_across_z_scale']

    data_loaders = dict()

    for idx, row in df.iterrows():
        print(idx, row.run_name)

        if row.dataset not in data_loaders:
            data_loaders[row.dataset] = get_test_validation_loader(dataset_name=row.dataset,
                                                                   batch_size=batch_size,
                                                                   num_workers=num_workers,
                                                                   include_train=include_train)

        # RESULT DIR + FILE
        save_dir = f"{EXPORT_DIR}/{row.run_name}"
        os.makedirs(save_dir, exist_ok=True)
        result_file = f"{save_dir}/{EVALUATION_RESULT_FILE}"

        if os.path.isfile(result_file) and not force_recompute:
            print("Did this one already, skipping it...")
            continue

        encode_file = f"{save_dir}/{ENCODING_FILE}"
        surprisal_file = f"{save_dir}/{SURPRISAL_DATA_FILE}"

        if not (os.path.exists(encode_file) or os.path.exists(surprisal_file)):
            print("No encodings or surprisal data found for model, re-run <encode_all> "
                  "and <gather_surprisal_stats> including this run.")
            continue

        results = dict()

        with torch.no_grad():

            encodings = torch.load(encode_file)
            surprisals = torch.load(surprisal_file)

            # CHECKPOINT
            checkpoint_path = f"{CHECKPOINT_DIR}/{row.run_name}.pt"
            vae_model, args = utils.load_checkpoint_model_for_eval(checkpoint_path, map_location=device,
                                                                   return_args=True)
            vae_model = vae_model.to(device)
            trainer = train.Trainer(args=args, dataset=None, vae_model=vae_model, device=device)

            # TRAIN / TEST / VALIDATE
            for phase, loader in data_loaders[row.dataset].items():
                results[phase] = dict()

                # MMD CALC
                if phase in encodings:
                    mmd = mmd_fn(encodings[phase]["z"], n_samples_mmd=n_samples_mmd)
                    results[phase]["MMD mean"] = mmd

                # ADD PPL & LL data
                if phase in surprisals:
                    results[phase]["PPL mean"] = surprisals[phase]["ppl"]
                    results[phase]["IW LL mean"] = np.mean(surprisals[phase]["iw_lls"])

                res = trainer.test(device=device, loader=loader)

                for k, v in res.items():
                    if k in save_metrics:
                        results[phase][k] = v
                        results[phase][k + " mean"] = np.mean(v)

            torch.save(results, result_file)

