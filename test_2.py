from arguments import prepare_parser
from vae_model.vae import VaeModel
from dataset_dataloader import ImageDataset
from train import Trainer
import torch

def main():
    config = prepare_parser(print_settings=True)

    config.batch_size = 3
    config.num_workers = 0
    config.latent_dim = 2

    config.image_or_language = "image"

    counter = 0

    config.objective = "VAE"

    # for dataset_name, data_dist in zip(["bmnist", "mnist"], ["bernoulli", "multinomial"]):
    dataset_name = "bmnist"
    data_dist = "bernoulli"
    print(dataset_name)
    config.image_dataset_name = dataset_name
    config.data_distribution = data_dist

    dataset = ImageDataset(args=config)
    train_loader = dataset.train_loader()

    for decoder_network_type in ["conditional_made_decoder", "basic_mlp_decoder", "basic_deconv_decoder"]: #, "conditional_made_decoder"
        config.decoder_network_type = decoder_network_type

        for encoder_network_type in ["basic_mlp_encoder", "basic_conv_encoder"]: # ,
            config.encoder_network_type = encoder_network_type

            for q_z_x_type in ["conditional_gaussian_made", "independent_gaussian"]: #, "independent_gaussian"

                config.q_z_x_type = q_z_x_type

                for p_z_type in ["isotropic_gaussian"]: #, "mog"
                    config.p_z_type = p_z_type
                    config.mog_n_components = 3

                    print(f"{counter} | dataset: {dataset_name.upper()}, data dist {data_dist}, encoder_network_type: {encoder_network_type}, decoder_network_type: {decoder_network_type}, q_z_x_type {q_z_x_type}, p_z_type {p_z_type}")
                    if decoder_network_type == "conditional_made_decoder" and data_dist == "multinomial":
                        print("This combo is not implemented yet.")
                        continue

                    vae = VaeModel(args=config)
                    trainer = Trainer(args=config, dataset=dataset, vae_model=vae, device="cpu")

                    with torch.autograd.set_detect_anomaly(True):
                        for i, (X, y) in enumerate(train_loader):
                            #print(f"{i} X input shape", X.shape)

                            #vae(X)
                            loss_dict = trainer.train_step(X)
                            #
                            #vae.estimate_log_likelihood(dataset.valid_loader())

                            # break

                            if i == 3:
                                break

                    print("\n\n")

                    counter += 1

if __name__ == "__main__":
    main()