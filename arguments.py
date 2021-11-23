import distutils
import configargparse
import sys
import os
import datetime
import yaml


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
    parser.add_argument("--free_bits", default=5.0, type=float, help="The number of Free bits.")
    parser.add_argument("--free_bits_per_dimension", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to apply the Free bits per dimension or with the dimensions combined.")

    # MDR-VAE
    parser.add_argument("--mdr_value", default=16.0, type=float,
                        help="The Minimum Desired Rate value (> constraint on rate).")
    parser.add_argument("--mdr_constraint_optim_lr", default=0.001, type=float,
                        help="The learning rate for the MDR constraint optimiser.")

    # INFO-VAE
    # parser.add_argument("--info_alpha", default=0.0, type=float,
    #                     help="The alpha parameter in the INFO-VAE objective (Zhao et al., 2017).")
    # parser.add_argument("--info_lambda", default=1000.0, type=float,
    #                     help="The lambda parameter in the INFO-VAE objective (Zhao et al., 2017).")
    parser.add_argument("--info_lambda_1_rate", default=1.0, type=float,
                        help="The lambda_1 parameter (for the Rate term) in the INFO-VAE objective as described in "
                             "the Lagrangian VAE paper.")
    parser.add_argument("--info_lambda_2_mmd", default=100.0, type=float,
                        help="The lambda_2 parameter (for the MMD term) in the INFO-VAE objective as described in "
                             "the Lagrangian VAE paper.")

    # LAG-INFO-VAE
    parser.add_argument("--rate_constraint_val", default=16.0, type=float,
                        help="The rate constraint value of the Lagrangian InfoVAE objective.")
    parser.add_argument("--rate_constraint_rel", default="eq", type=str,
                        help="The relation for the rate constraint in the Lagrangian InfoVAE objective, options:"
                             "- 'ge' = '>='"
                             "- 'le' = '<='"
                             "- 'eq' = '='")
    parser.add_argument("--rate_constraint_lr", default=0.001, type=float,
                        help="The rate constraint optimiser learning rate.")

    parser.add_argument("--mmd_constraint_val", default=0.005, type=float,
                        help="The MMD constraint value of the Lagrangian InfoVAE objective.")
    parser.add_argument("--mmd_constraint_rel", default="le", type=str,
                        help="The relation for the MMD constraint in the Lagrangian InfoVAE objective, options:"
                             "- 'ge' = '>='"
                             "- 'le' = '<='"
                             "- 'eq' = '='")
    parser.add_argument("--mmd_constraint_lr", default=0.001, type=float,
                        help="The MMD constraint optimiser learning rate.")

    # ----------------------------------------------------------------------------------------------------------------
    # BATCHES / TRAIN STEPS
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for data loading and training.")
    parser.add_argument("--max_steps", default=int(1e6), type=int,
                        help="Maximum number of train steps in total.")
    parser.add_argument("--max_epochs", default=120, type=int,
                        help="Maximum number of epochs, if -1 no maximum number of epochs is set.")
    parser.add_argument("--eval_ll_every_n_epochs", default=1, type=int,
                        help="Every how many epochs to evaluate importance weighted log likelihood.")

    # ----------------------------------------------------------------------------------------------------------------
    # OPTIMISATION
    parser.add_argument("--gen_opt", type=str, default="adam")
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--gen_l2_weight", type=float, default=1e-4)
    parser.add_argument("--gen_momentum", type=float, default=0.0)

    parser.add_argument("--inf_opt", type=str, default="adam")
    parser.add_argument("--inf_lr", type=float, default=1e-4)
    parser.add_argument("--inf_l2_weight", type=float, default=1e-4)
    parser.add_argument("--inf_momentum", type=float, default=0.0)

    parser.add_argument("--max_gradient_norm", type=float, default=1.0)
    # TODO: parser.add_argument("--lr_scheduler", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
    #                     help="Whether or not to use a lr scheduler (default: True).")
    parser.add_argument("--iw_n_samples", type=int, default=50,
                        help="Number of samples for IW LL estimation.")

    # ----------------------------------------------------------------------------------------------------------------
    # ARCHITECTURE
    parser.add_argument("--latent_dim", default=32, type=int, help="Dimensionality of the latent space.")
    parser.add_argument("--decoder_network_type", default="basic_deconv_decoder", type=str,
                        help="Which architecture / distribution structure to use for decoder, options:"
                             "  - basic_mlp_decoder (image)"
                             "  - basic_deconv_decoder (image):"
                             "      p(x|z)"
                             "  - conditional_made_decoder (image),"
                             "  - cond_pixel_cnn_pp (image):"
                             "      p(x_d|z, x<d)"
                             "  - distil_roberta_strong_decoder (language)"
                             "  - distil_roberta_weak_decoder (language)")
    parser.add_argument("--decoder_MADE_gating", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to make use of (learned) gated addition of the context.")
    parser.add_argument("--decoder_MADE_gating_mechanism", default=0, type=int,
                        help="What gating mechanism to use for adding the latent as context:"
                             "  - 0: learned gate: h = act(t(h) + c(context) * sigmoid(gate_i))"
                             "  - 1: pixel cnn like gate: h = act( m_lin(h) + lin(z)) * sigmoid(m_lin(h) + lin(z))")
    parser.add_argument("--decoder_MADE_hidden_sizes", default="200-220", type=str,
                        help="Sizes of the hidden layers of the decoder MADE, format as H-H-H")
    parser.add_argument("--encoder_network_type", default="basic_conv_encoder", type=str,
                        help="Which architecture / distribution structure to use for decoder, options:"
                             "  - basic_mlp_encoder (image)"
                             "  - basic_conv_encoder (image)"
                             "  - distil_roberta_encoder (language)")
    parser.add_argument("--encoder_MADE_hidden_sizes", default="200-220", type=str,
                        help="Sizes of the hidden layers of the decoder MADE, format as H-H-H")

    # ----------------------------------------------------------------------------------------------------------------
    # DISTRIBUTION TYPES
    # independent_gaussian", conditional_gaussian_made, iaf
    parser.add_argument("--q_z_x_type", default="independent_gaussian", type=str,
                        help="Which type of posterior distribution to use, options:"
                             "  - independent_gaussian"
                             "  - conditional_gaussian_made"
                             "  - standard_normal"
                             "  - iaf")
    # isotropic_gaussian, ...
    parser.add_argument("--p_z_type", default="isotropic_gaussian", type=str,
                        help="Which type of prior distribution to use, options:"
                             "  - isotropic_gaussian"
                             "  - mog")
    parser.add_argument("--mog_n_components", default=10, type=int,
                        help="If using Mixture of Gaussians as prior, "
                             "this parameter sets the number of learned components.")
    # p_x_z_type: [bernoulli, gaussian, multinomial]
    parser.add_argument("--p_x_z_type", default="bernoulli", type=str,
                        help="Which type of predictive p_x_z distribution to use, options:"
                             "  - bernoulli"
                             "  - gaussian"
                             "  - multinomial")

    # ----------------------------------------------------------------------------------------------------------------
    # GENERAL DATASET ARGUMENTS
    parser.add_argument("--data_dir", default=get_code_dir() + '/data', type=str,
                        help="The name of the data directory.")
    parser.add_argument("--image_or_language", default='language', type=str,
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
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str,
                        help="The name of the tokenizer, 'roberta-base' by default.")
    parser.add_argument("--language_dataset_name", default='ptb', type=str,
                        help="The name of the dataset, 'yahoo_answer' by default, options:"
                             "  - yahoo_answer"
                             "  - ptb")
    parser.add_argument("--vocab_size", default=50265, type=int,
                        help="Size of the vocabulary size of the tokenizer used.")  # 50265 = roberta vocab size
    parser.add_argument("--num_workers", default=6, type=int,
                        help="Num workers for data loading.")
    parser.add_argument("--pin_memory", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not pin memory (set to True if use gpu).")
    parser.add_argument("--max_seq_len", default=64, type=int,
                        help="What the maximum sequence length the model accepts is (default: 128).")

    # ----------------------------------------------------------------------------------------------------------------
    # PRINTING & LOGGING
    parser.add_argument("--print_stats", default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not print stats.")
    parser.add_argument("--print_every_n_steps", default=1, type=int,
                        help="Every how many steps to print.")
    parser.add_argument("--logging", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to log the process of the model (default: True).")
    parser.add_argument("--checkpointing", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to save checkpoints.")
    parser.add_argument("--log_every_n_steps", default=1, type=int,
                        help="Every how many steps to log (default: 20).")
    parser.add_argument("--wandb_project", default='fall-2021-VAE', type=str,
                        help="The name of the W&B project to store runs to.")

    # ----------------------------------------------------------------------------------------------------------------
    # CHECKPOINTING
    # parser.add_argument("--checkpoint", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
    #                     help="Whether or not to checkpoint (save) the model. (default: False).")
    # parser.add_argument("--checkpoint_every_n_steps", default=10, type=int,
    #                     help="Every how many (training) steps to checkpoint (default: 1000).")

    # ----------------------------------------------------------------------------------------------------------------
    # DISTRIBUTED TRAINING
    parser.add_argument("--gpus", default=0, type=int,
                        help="Number GPUs to use (default: None).")
    parser.add_argument("--ddp", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="Whether or not to use Distributed Data Parallel (DDP) "
                             "(default: True if n_gpus > 1, else: False).")

    # ----------------------------------------------------------------------------------------------------------------
    # GENERAL
    parser.add_argument("--code_dir", default=get_code_dir(), type=str,
                        help="The name of the code dir, depending on LISA or local.")
    parser.add_argument("--job_id", default="", type=str,
                        help="...")
    parser.add_argument("--short_dev_run", default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help="...")
    parser.add_argument("--run_name_prefix", default="", type=str,
                        help="A prefix to pre-pend to the run_name argument.")
    parser.add_argument("--run_name", default="", type=str,
                        help="Run name that will show up in W&B.")
    parser.add_argument("--checkpoint_dir", default=get_code_dir()+"/run_files/checkpoints/", type=str,
                        help="Which directories to save checkpoints at.")
    parser.add_argument("--wandb_dir", default=get_code_dir() + "/run_files/", type=str,
                        help="Which directories to save checkpoints at.")



    # TODO: add seed & deterministic

    if jupyter:
        sys.argv = [sys.argv[0]]

    args = parser.parse_args()

    os.makedirs(get_code_dir()+"/run_files", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.run_name = make_run_name(args)

    check_settings(args)

    if print_settings: print_args(args)

    return args


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


def get_code_dir():
    return os.path.dirname(os.path.realpath(__file__))


def make_run_name(args):
    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    date, time = datetime_stamp.split("--")[0], datetime_stamp.split("--")[1]
    date_time = f"{date}-{time}"

    if args.objective == "MDR-VAE":
        obj = f"MDR-VAE[R>={args.mdr_value}]"

    elif args.objective == "FB-VAE":
        obj = f"FB-VAE[{args.free_bits}-pd={args.free_bits_per_dimension}]"

    elif args.objective == "BETA-VAE":
        obj = f"B-VAE[b={args.beta_beta}]"

    elif args.objective == "INFO-VAE":
        obj = f"INFO-VAE[l_1_rate={args.info_lambda_1_rate}, l_2_mmd={args.info_lambda_2_mmd}]"

    elif args.objective == "LAG-INFO-VAE":
        obj = f"LAG-INFO-VAE[rate-{args.rate_constraint_rel}-{args.rate_constraint_val}, mmd-{args.mmd_constraint_rel}-{args.mmd_constraint_val}]"

    else:
        obj = args.objective

    name = f"{obj} | q(z|x) {args.q_z_x_type} | p(x|z) {args.decoder_network_type} | p(z) {args.p_z_type} | D = {args.latent_dim} | {date_time}"

    # name = f"dec_made_hs {args.decoder_MADE_hidden_sizes} p(z) {args.p_z_type} D {args.latent_dim} | {date_time}"

    return args.run_name_prefix + name


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
    objective_options = ["VAE", "AE", "FB-VAE", "BETA-VAE", "MDR-VAE", "INFO-VAE", "LAG-INFO-VAE"]
    check_valid_option(args.objective, objective_options, "objective")

    # Decoder network types
    decoder_network_type_options = ["basic_mlp_decoder", "basic_deconv_decoder", "conditional_made_decoder",
                                    "cond_pixel_cnn_pp", "distil_roberta_strong_decoder", "distil_roberta_weak_decoder"]
    check_valid_option(args.decoder_network_type, decoder_network_type_options, "decoder_network_type")

    MADE_gating_mech_options = [0, 1]
    check_valid_option(args.decoder_MADE_gating_mechanism, MADE_gating_mech_options, "decoder_MADE_gating_mechanism")

    # Encoder network types
    encoder_network_type_options = ["basic_mlp_encoder", "basic_conv_encoder", "distil_roberta_encoder"]
    check_valid_option(args.encoder_network_type, encoder_network_type_options, "encoder_network_type")

    # Posterior types
    q_z_x_type_options = ["independent_gaussian", "conditional_gaussian_made", "iaf", "standard_normal"]
    check_valid_option(args.q_z_x_type, q_z_x_type_options, "q_z_x_type")

    if args.q_z_x_type == "iaf":
        raise NotImplementedError

    # Prior type
    p_z_type_options = ["isotropic_gaussian", "mog"]
    check_valid_option(args.p_z_type, p_z_type_options, "p_z_type")

    p_x_z_type_options = ["bernoulli", "multinomial", "categorical"]
    check_valid_option(args.p_x_z_type, p_x_z_type_options, "p_x_z_type")

    if args.decoder_network_type == "conditional_made_decoder" and not (args.p_x_z_type == "bernoulli"):
        raise NotImplementedError


if __name__ == "__main__":
    config = prepare_parser(jupyter=False, print_settings=True)

    default_config_file = "test_config.yaml"
    print("Dumping default config in:", default_config_file)
    with open(default_config_file, 'w') as file:
        documents = yaml.dump(vars(config), file)
