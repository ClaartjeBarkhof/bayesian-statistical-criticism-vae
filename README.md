# File structure

# `/analysis`
  - `/bda_models`
    - `bda_dp_mixture_surprisal_vals.py`: 
        analyse surprisal values with DP Mixture (NumPyro model + plot functions)
    - `bda_MM_latent_analysis.py`: 
        analyse latent samples with DP Mixture (calls Wilker's Pyro Model)
    - `bda_pixel_model_mnist.py`: 
        conditional beta bernoulli pixel model (NumPyro + plot functions)
    - `bda_sequence_length_model_ptb.py`: 
        rate poisson model (NumPyro + plot functions)
    - `bda_topic_model_ptb.py`
        topic model (uses altered Gensim LDA implementation, optimised with VI)
    - `gensim_LDA.py`
        alteration of the class by Gensim
    - `/Pyro_BDA`: holds code from `probabll/bda` repository
  - `/data_space`:
    - `MNIST_pixels.ipynb`: fit BDA & compute surprisals for MNIST
    - `PTB_sequence_length_preprocess.ipynb`: pre-process for length analyis (outputs `ptb_length_analysis_data.pt`)
    - `PTB_sequence_length.ipynb`: fit BDA & compute surprisals for PTB sequence length analysis
    - `PTB_topics.ipynb`: fit BDA & compute surprisals for PTB lda topic analysis
    - `surprisal_DPs.ipynb`: fit DPs on surprisal values
    - `/output_files`: stores intermediate output files
  - `/latent_space`
    - `latent_space_analysis.ipynb`: perform latent space analysis
    - `latent_analysis.py`: some functions that help in `latent_space_analysis.ipynb`
    - `image_encoding_stats.csv`: stores some basic statistics of the image encodings (KS, MMD, etc.)
    - `language_encoding_stats.csv`: stores some basic statistics of the language (KS, MMD, etc.)
  - final_selection_runs.csv: file that stores the experiments used in the analysis
  - global_stats.csv: file that stores aggregated intrinsic evaluation results used in the analysis
