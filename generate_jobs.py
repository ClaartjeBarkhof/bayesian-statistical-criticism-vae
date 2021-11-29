import slurmjobs
import os
import shutil

batch = slurmjobs.SlurmBatch(
    'python /home/cbarkhof/fall-2021/train.py')

# generate jobs across parameter grid
run_script, job_paths = batch.generate([
    ("beta_beta", [0.0, 0.1, 0.5, 1.0, 1.5]),
    ("decoder_network_type", ["weak_memory_distil_roberta_decoder", "strong_distil_roberta_decoder"])],
    batch_size=64,
    # beta_beta=0.1,
    latent_dim=128,
    # mdr_value=4.0,
    # mdr_constraint_optim_lr=0.001,
    checkpointing=True,
    data_distribution="categorical",
    # decoder_MADE_gating=True,
    # decoder_MADE_gating_mechanism=0,
    # decoder_MADE_hidden_sizes=200-220,
    # decoder_network_type="cond_pixel_cnn_pp",
    # encoder_MADE_hidden_sizes=200-220,
    encoder_network_type="distil_roberta_encoder",
    eval_ll_every_n_epochs=1000,
    # free_bits=5.0,
    # free_bits_per_dimension=False,
    gen_l2_weight=0.0001,
    gen_lr=0.00001,
    gen_momentum=0.0,
    gen_opt="adam",
    gpus=1,
    # image_dataset_name="bmnist",
    image_or_language="language",
    # image_w_h=28,
    inf_l2_weight=0.0001,
    inf_lr=0.00001,
    inf_momentum=0.0,
    inf_opt="adam",
    # info_lambda_1_rate=0.5,
    # info_lambda_2_mmd=1.0,
    iw_n_samples=1,
    language_dataset_name="ptb",
    # latent_dim=768,
    log_every_n_steps=5,
    logging=True,
    max_epochs=120,
    max_gradient_norm=1.0,
    max_seq_len=64,
    max_steps=1000000,
    # mdr_constraint_optim_lr=0.001,
    # mdr_value=16.0,
    # mmd_constraint_lr=0.001,
    # mmd_constraint_rel="le",
    # mmd_constraint_val=0.001,
    # mog_n_components=5,
    # n_channels=1,
    num_workers=3,
    objective="BETA-VAE",
    pin_memory=True,
    p_z_type="isotropic_gaussian",
    print_every_n_steps=50,
    print_stats=True,
    q_z_x_type="independent_gaussian",
    # rate_constraint_lr=0.001,
    # rate_constraint_rel="ge",
    # rate_constraint_val=16.0,
    run_name_prefix='(29-nov ptb-beta-vae) ',
    short_dev_run=False,
    strong_roberta_decoder_embedding_dropout=False,
    tokenizer_name="roberta-base",
    vocab_size=50265,
    wandb_project="fall-2021-VAE")

slurmjobs.util.summary(run_script, job_paths)

basic_stuff = """#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 03:00:00
#SBATCH --mem 10G
#SBATCH --output /home/cbarkhof/slurm-logs/%j-slurm-log.out

module purge  # unload all that are active
module load 2019  # load 2019 software module for good python versions
module load Anaconda3/2018.12  # load anacoda
module load CUDA/10.0.130  # load cuda
module load cuDNN/7.6.3-CUDA-10.0.130  # load cudnn
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.6.3-CUDA-10.0.130/lib64:$LD_LIBRARY_PATH

CONDA_PREFIX=$(conda info --base)
source $CONDA_PREFIX/etc/profile.d/conda.sh

conda deactivate # just to make sure other envs are not active
conda activate thesisenv # activate environment

"""

job_dir = "jobs/home.cbarkhof.fall-2021.train"
for i, f in enumerate(os.listdir(job_dir)):

    if ".sbatch" in f or ".sh" in f:

        # Read in the file
        with open(f'{job_dir}/{f}', 'r') as file:
            filedata = file.read()

            idx = filedata.find("# run script with arguments")
            length = len("# run script with arguments")
            start_idx = int(idx + length + 1)
            command = filedata[start_idx:]

            newfiledata = basic_stuff + command

            with open(f'{job_dir}/{f}', 'w') as file:
                file.flush()
                file.write(newfiledata)

            os.rename(f'{job_dir}/{f}', f'{job_dir}/{f}'.replace(",", "-"))

# Remove files I don't need
os.remove("jobs/home.cbarkhof.fall-2021.train/run.sh")
os.rmdir("jobs/home.cbarkhof.fall-2021.train/slurm")
os.remove("jobs/home.cbarkhof.fall-2021.train/time_generated")
for fd in os.listdir("jobs/"):
    if "~" in fd:
        p = f"jobs/{fd}"
        print(p)
        shutil.rmtree(p)

for i, f in enumerate(os.listdir(job_dir)):
    if ".sbatch" in f:
        with open("jobs/home.cbarkhof.fall-2021.train/run.sh", "a") as file_object:
            file_object.write(f"sbatch '{job_dir}/{f}'\n")