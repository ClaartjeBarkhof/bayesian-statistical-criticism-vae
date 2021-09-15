from arguments import prepare_parser
from vae_model.vae import VaeModel
from dataset_dataloader import ImageDataset

def main():
    config = prepare_parser(print_settings=True)

    config.batch_size = 3
    config.latent_dim = 2


    config.image_or_language = "image"

    counter = 0
    config.decoder_network_type = "basic_decoder"
    # config.q_z_x_type = "conditional_gaussian_made"  # "iaf" #"independent_gaussian" # conditional_gaussian_made

    for dataset_name, data_dist in zip(["bmnist", "fmnist", "mnist"], ["bernoulli", "multinomial", "multinomial"]):

        config.image_dataset_name = dataset_name
        config.data_distribution = data_dist

        dataset = ImageDataset(args=config)
        train_loader = dataset.train_loader()

        for q_z_x_type in ["conditional_gaussian_made", "independent_gaussian"]:

            config.q_z_x_type = q_z_x_type

            for p_z_type in ["isotropic_gaussian", "mog"]:
                config.p_z_type = p_z_type
                config.mog_n_components = 3

                print(f"{counter} | dataset: {dataset_name.upper()}, data dist {data_dist}, q_z_x_type {q_z_x_type}, p_z_type {p_z_type}")
                vae = VaeModel(args=config)

                for X, y in train_loader:
                    print("X input shape", X.shape)
                    #vae(X)

                    vae.training_step((X, y), 0)

                    break

                sample = vae.gen_model.sample_generative_model(S=10)
                print("Sample.shape", sample.shape)


                print("\n\n")

                counter += 1

if __name__ == "__main__":
    main()