from arguments import prepare_parser, check_settings
from vae_model.vae import VaeModel
from dataset_dataloader import ImageDataset
from train import Trainer
import torch
import train

def test_main():
    config = prepare_parser(print_settings=False)

    # Simple settings
    config.batch_size = 3
    config.num_workers = 0
    config.latent_dim = 4

    # Logging checkpointing testing
    config.run_name_prefix = "TEST __ "
    config.short_dev_run = True
    config.logging = True
    config.checkpointing = True

    # Simple objective
    config.objective = "VAE"

    # DATA
    config.image_or_language = "image"
    config.image_dataset_name = "bmnist"
    config.data_distribution = "bernoulli"
    config.num_workers = 0

    counter = 0

    for decoder_network_type in ["basic_mlp_decoder", "conditional_made_decoder",  "basic_deconv_decoder"]:
        config.decoder_network_type = decoder_network_type

        for encoder_network_type in ["basic_mlp_encoder", "basic_conv_encoder"]:
            config.encoder_network_type = encoder_network_type

            for q_z_x_type in ["independent_gaussian", "conditional_gaussian_made"]:
                config.q_z_x_type = q_z_x_type

                for p_z_type in ["isotropic_gaussian", "mog"]:
                    config.p_z_type = p_z_type
                    config.mog_n_components = 3

                    check_settings(config)

                    print(f"{counter} | dataset: {config.image_dataset_name.upper()}, data dist {config.data_distribution}, "
                          f"encoder_network_type: {encoder_network_type}, decoder_network_type: {decoder_network_type}, "
                          f"q_z_x_type {q_z_x_type}, p_z_type {p_z_type}")

                    if decoder_network_type == "conditional_made_decoder" and config.data_distribution == "multinomial":
                        print("This combo is not implemented yet.")
                        continue

                    train.main(config=config)

                    counter += 1


if __name__ == "__main__":
    test_main()
