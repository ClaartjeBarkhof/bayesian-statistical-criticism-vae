import slurmjobs
import os
import shutil

batch = slurmjobs.SlurmBatch(
    'python /home/cbarkhof/fall-2021/train.py')

# generate jobs across parameter grid
run_script, job_paths = batch.generate([
    # ('p_z_type', ['isotropic_gaussian', 'mog']),
    ('q_z_x_type', ["independent_gaussian", "conditional_gaussian_made"]),
    ('encoder_network_type', ["basic_conv_encoder", "basic_mlp_encoder"]),
    ('decoder_network_type', ['basic_mlp_decoder', 'basic_deconv_decoder', 'conditional_made_decoder'])],
    p_z_type="isotropic_gaussian",
    objective="VAE",
    beta_beta=0.9,
    mdr_value=8.0,
    mdr_constraint_optim_lr=0.001,
    info_alpha=0.0,
    info_lambda=1000.0,
    batch_size=64,
    max_epochs=120,
    eval_ll_every_n_epochs=1,
    latent_dim=10,
    mog_n_components=10,
    p_x_z_type="bernoulli",
    data_distribution="bernoulli",
    image_or_language="image",
    image_dataset_name="bmnist",
    print_stats=True,
    print_every_n_steps=50,
    iw_n_samples=50,
    logging=True,
    log_every_n_steps=5,
    short_dev_run=False,
    checkpointing=True,
    gpus=1)

slurmjobs.util.summary(run_script, job_paths)

basic_stuff = """#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 03:00:00
#SBATCH --mem 10G
#SBATCH --output /home/cbarkhof/slurm-logs/%j-slurm-log.out

module purge  # unload all that are active
module load 2019  # load 2019 software module for good python versions
module load Anaconda3  # load anacoda
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