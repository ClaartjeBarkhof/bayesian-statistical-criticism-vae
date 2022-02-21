import sys
sys.path.append("/home/cbarkhof/fall-2021")
sys.path.append("/home/cbarkhof/fall-2021/analysis_run_postprocessing")

from general_postprocessing_steps import *

DEVICE = "cuda:0"

prefixes = ["(20-jan bmnist)"]

run_df = make_run_overview_df(prefixes=prefixes)

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# 20 x 100 = 2000 (un)conditional samples
# encode all batches
encode_sample_all(run_df, device=DEVICE, include_train=True, n_sample_batches=10,
                  sample_batch_size=100, tok=tokenizer, batch_size=128)

with torch.no_grad():
    # batch size surprisal is low as the effective batch is (batch_size_surprisal * n_iw_samples))
    gather_surprisal_stats(run_df, device=DEVICE, include_train=True,
                           batch_size_surprisal=4, max_batches=125, n_iw_samples=50)

with torch.no_grad():
    evaluate(run_df, include_train=True, n_samples_mmd=5000, num_workers=3, batch_size=128, device=DEVICE)