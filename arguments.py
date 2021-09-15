import distutils
import configargparse
import sys


def prepare_parser(jupyter=False, print_settings=True):
    parser = configargparse.ArgParser()
    parser.add_argument('--config_file', required=False, is_config_file=True, help='config file path')

    # TODO: possibly add AMP, early stopping, load from checkpoint, etc.

    # ----------------------------------------------------------------------------------------------------------------
    # OBJECTIVE
    parser.add_argument("--objective", default="VAE", type=str,
                        help="Which objective to use, options:"
                             "  - VAE"
                             "  - AE"
                             "  - BETA-VAE, with beta argument set (Higgins et al., 2016)"
                             "  - FB-VAE (Kingma et al., 2016)"
                             "  - MDR-VAE (Pelsmaeker & Aziz, 2019)"
                             "  - INFO-VAE, with alpha and lambda argument set  (Zhao et al., 2017)"
                             "  - LAG-INFO-VAE (with ...)  (Zhao et al., 2017)")
    # BETA-VAE
    parser.add_argument("--beta_beta", default=1.0, type=float,
                        help="Beta in BETA-VAE objective.")

    # FB-VAE
    parser.add_argument("--free_bits", default=0.5, type=float, help="The number of Free bits.")
    parser.add_argument("--free_bits_per_dimension", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to apply the Free bits per dimension or with the dimensions combined.")

    # MDR-VAE
    parser.add_argument("--mdr_value", default=16.0, type=float,
                        help="The Minimum Desired Rate value (> constraint on rate).")

    # INFO-VAE
    parser.add_argument("--info_alpha", default=1.0, type=float,
                        help="The alpha parameter in the INFO-VAE objective (Zhao et al., 2017).")
    parser.add_argument("--info_lambda", default=1000.0, type=float,
                        help="The lambda parameter in the INFO-VAE objective (Zhao et al., 2017).")

    # ----------------------------------------------------------------------------------------------------------------
    # BATCHES / TRAIN STEPS
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--max_global_train_steps", default=1e6, type=int,
                        help="Maximum number of train steps in total.")
    parser.add_argument("--max_epochs", default=100, type=int,
                        help="Maximum number of epochs, if -1 no maximum number of epochs is set.")

    # ----------------------------------------------------------------------------------------------------------------
    # LEARNING RATE
    parser.add_argument("--lr", default=0.00005, type=float,
                        help="Learning rate (default: 0.00002).")
    parser.add_argument("--lr_scheduler", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use a lr scheduler (default: True).")

    # ----------------------------------------------------------------------------------------------------------------
    # ARCHITECTURE
    parser.add_argument("--latent_dim", default=32, type=int, help="Dimensionality of the latent space.")
    parser.add_argument("--decoder_network_type", default="basic_deconv_decoder", type=str,
                        help="Which architecture / distribution structure to use for decoder, options:"
                             "  - basic_deconv_decoder:"
                             "      p(x|z)"
                             "  - conditional_made_decoder:"
                             "      p(x_d|z, x<d)")
    # ----------------------------------------------------------------------------------------------------------------
    # DISTRIBUTION TYPES
    # independent_gaussian, ...
    parser.add_argument("--q_z_x_type", default="independent_gaussian", type=str,
                        help="Which type of posterior distribution to use, options:"
                             "  - independent_gaussian"
                             "  - ...")
    # isotropic_gaussian, ...
    parser.add_argument("--p_z_type", default="isotropic_gaussian", type=str,
                        help="Which type of prior distribution to use, options:"
                             "  - isotropic_gaussian"
                             "  - mog")
    parser.add_argument("--mog_n_components", default=20, type=int,
                        help="If using Mixture of Gaussians as prior, "
                             "this parameter sets the number of learned components.")
    # p_x_z_type: [bernoulli, gaussian, categorical]
    parser.add_argument("--p_x_z_type", default="bernoulli", type=str,
                        help="Which type of predictive p_x_z distribution to use, options:"
                             "  - bernoulli"
                             "  - gaussian"
                             "  - categorical")

    # ----------------------------------------------------------------------------------------------------------------
    # GENERAL DATASET ARGUMENTS
    parser.add_argument("--data_dir", default='data', type=str, help="The name of the data directory.")
    parser.add_argument("--image_or_language", default='image', type=str,
                        help="The type of the dataset, options: 'image' or 'language'.")
    parser.add_argument("--data_distribution", default='bernoulli', type=str,
                        help="The type of data distribution, bernoulli for binary inputs and"
                             "multinomial for categorical inputs.")

    # ----------------------------------------------------------------------------------------------------------------
    # IMAGE DATASET ARGUMENTS
    parser.add_argument("--image_dataset_name", default='bmnist', type=str, help="The name of the image dataset.")
    parser.add_argument("--image_w_h", default=28, type=int, help="The width and height of the (square) "
                                                                  "images in the data set.")
    parser.add_argument("--n_channels", default=1, type=int, help="The name of the image dataset.")

    # ----------------------------------------------------------------------------------------------------------------
    # LANGUAGE DATASET ARGUMENTS
    parser.add_argument("--tokenizer_name", default='roberta', type=str,
                        help="The name of the tokenizer, 'roberta' by default.")
    parser.add_argument("--language_dataset_name", default='ptb_text_only', type=str,
                        help="The name of the dataset, 'cnn_dailymail' by default, else: ptb_text_only.")
    parser.add_argument("--vocab_size", default=50265, type=int,
                        help="Size of the vocabulary size of the tokenizer used.") # 50265 = roberta vocab size
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Num workers for data loading.")
    parser.add_argument("--max_seq_len", default=64, type=int,
                        help="What the maximum sequence length the model accepts is (default: 128).")

    # ----------------------------------------------------------------------------------------------------------------
    # PRINTING & LOGGING
    parser.add_argument("--print_stats", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not print stats.")
    parser.add_argument("--print_every_n_steps", default=10, type=int,
                        help="Every how many steps to print.")
    parser.add_argument("--logging", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to log the process of the model (default: True).")
    parser.add_argument("--log_every_n_steps", default=10, type=int,
                        help="Every how many steps to log (default: 20).")
    parser.add_argument("--wandb_project", default='fall-2021-VAE', type=str,
                        help="The name of the W&B project to store runs to.")

    # ----------------------------------------------------------------------------------------------------------------
    # CHECKPOINTING
    parser.add_argument("--checkpoint", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to checkpoint (save) the model. (default: False).")
    parser.add_argument("--checkpoint_every_n_steps", default=10, type=int,
                        help="Every how many (training) steps to checkpoint (default: 1000).")

    # ----------------------------------------------------------------------------------------------------------------
    # DISTRIBUTED TRAINING
    parser.add_argument("--n_gpus", default=1, type=int,
                        help="Number GPUs to use (default: None).")
    parser.add_argument("--ddp", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use Distributed Data Parallel (DDP) "
                             "(default: True if n_gpus > 1, else: False).")

    # ----------------------------------------------------------------------------------------------------------------
    # PARSE & PRINT & RETURN

    # TODO: add seed & deterministic

    if jupyter:
        sys.argv = [sys.argv[0]]

    args = parser.parse_args()

    check_settings(args)

    if print_settings: print_args(args)

    return parser.parse_args()


def print_args(args):
    print("-" * 71)
    print("-" * 30, "ARGUMENTS", "-" * 30)
    print("-" * 71)

    for k, v in vars(args).items():
        print(k, ":", v)

    print("-" * 70)
    print("-" * 70)


def check_valid_option(option, options, setting):
    assert option in options, f"{option} not a valid option for {setting}, valid options: {options}"


def check_settings(args):
    # DATA DISTRIBUTION / DATA SET CHOICES
    valid_dists = ["multinomial", "bernoulli"]
    assert args.data_distribution in valid_dists, \
        f"Invalid data distribution: {args.data_distribution}, must be one of: {valid_dists}"
    assert not (args.image_dataset_name == "bmnist" and args.data_distribution == "multinomial"), \
        f"If the data set is Binarised MNIST, the data distribution should be set to bernoulli, " \
        f"currently set to {args.data_distribution}."
    assert not (args.image_dataset_name in ["fmnist", "mnist"] and args.data_distribution == "bernoulli"), \
        f"If the data set is MNIST or Fashion MNIST, the data distribution should be set to " \
        f"multinomial, currently set to {args.data_distribution}."
    assert not (args.image_dataset_name in ["bminst", "fmnist", "mnist"] and not args.n_channels == 1), \
        f"{args.image_dataset_name} is a 1-channel dataset."

    # Objective
    objective_options = ["VAE", "AE", "FB-VAE", "MDR-VAE", "INFO-VAE", "LAG-INFO-VAE"]
    check_valid_option(args.objective, objective_options, "objective")

    # Decoder network types
    decoder_network_type_options = ["basic_deconv_decoder", "conditional_made_decoder"]
    check_valid_option(args.decoder_network_type, decoder_network_type_options, "decoder_network_type")

    # Posterior types
    q_z_x_type_options = ["independent_gaussian", "conditional_gaussian_made", "iaf"]
    check_valid_option(args.q_z_x_type, q_z_x_type_options, "q_z_x_type")

    # Prior type
    p_z_type_options = ["isotropic_gaussian", "mog"]
    check_valid_option(args.p_z_type, p_z_type_options, "p_z_type")


