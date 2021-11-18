# ------------------------------------------------------------------------------------------------
# IMPORTS
import sys; sys.path.append("/home/cbarkhof/fall-2021")
import torch
import argparse
import distutils

from analysis.analysis_steps import make_run_overview_df
from analysis.analysis_steps import encode_reconstruct_sample
from analysis.analysis_steps import gather_surprisal_stats
from analysis.analysis_steps import simple_evaluate_valid_test
from analysis.analysis_steps import knn_predictions_for_samples_reconstructions

# ------------------------------------------------------------------------------------------------
# GLOBALS
DEVICE = f"cuda:{torch.cuda.current_device()}"

CODE_DIR = "/home/cbarkhof/fall-2021"
PLOT_DIR = f"{CODE_DIR}/notebooks/plots"
ANALYSIS_DIR = f"{CODE_DIR}/analysis/analysis-files"
CHECKPOINT_DIR = f"{CODE_DIR}/run_files/checkpoints"

ENCODE_RECONSTUCT_FILE = f"encode-reconstruct-test-valid.pt"
SAMPLE_FILE = f"generative-samples.pt"

SURPRISAL_RECONSTRUCT_FILE = "surprisal_reconstruct.pt"
SURPRISAL_SAMPLE_FILE = "surprisal_sample.pt"
SURPRISAL_DATA_FILE = "surprisal_data.pt"

TEST_VALID_EVAL_FILE = "test-valid-results.pt"

KNN_PREDICT_SAMPLES_FILE = "knn-preds-generative-samples.pickle"
KNN_PREDICT_RECONSTRUCTIONS_FILE = "knn-preds-reconstructions.pickle"

KNN_PREDICT_STATS_FILE = "knn-preds-stats.pickle"

DATA_SPACE_STATS = "data_space_stats.pickle"

N_SAMPLE_BATCHES = 5
SAMPLE_BATCH_SIZE = 250

BATCH_SIZE_SURPRISAL = 30
N_IW_SAMPLES = 50

KNN_BATCH_SIZE = 100
BATCH_SIZE = 100
NUM_WORKERS = 3

# Whether or not to progress in reversed order
parser = argparse.ArgumentParser()
parser.add_argument('--reversed', required=True, type=lambda x: bool(distutils.util.strtobool(x)))
args = parser.parse_args()
REVERSE = args.reversed

# ------------------------------------------------------------------------------------------------
# Step 0: Retrieve runs to process
prefixes = ["(mdr-vae-exp 8 oct)", "(fb-vae-exp 8 oct) ", "(beta-vae-exp 6 oct) ", "(inf-vae-exp 5 oct) "]
run_df = make_run_overview_df(prefixes=prefixes)
# run_df.drop("run_name", axis=1) for in NBs

# ------------------------------------------------------------------------------------------------
# Step 1: Encode, reconstruct, sample
encode_reconstruct_sample(run_df, device=DEVICE, include_train=True,
                          n_sample_batches=N_SAMPLE_BATCHES,
                          sample_batch_size=SAMPLE_BATCH_SIZE, reverse=REVERSE)

# ------------------------------------------------------------------------------------------------
# Step 2: Gather surprisal stats
gather_surprisal_stats(device=DEVICE, include_train=True,
                       batch_size_surprisal=BATCH_SIZE_SURPRISAL, n_iw_samples=N_IW_SAMPLES, reverse=REVERSE)

# ------------------------------------------------------------------------------------------------
# Step 3: Gather simple evaluation of validation and test set
# simple_evaluate_valid_test(run_df, device=DEVICE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# ------------------------------------------------------------------------------------------------
# Step 4: KNN predictions of samples & reconstructions
# knn_predictions_for_samples_reconstructions(batch_size=KNN_BATCH_SIZE,
#                                             knn_mimicker_path="/home/cbarkhof/fall-2021/notebooks"
#                                                               "/KNN_mimicking_network.pt",
#                                             knn_path=None, device=DEVICE)