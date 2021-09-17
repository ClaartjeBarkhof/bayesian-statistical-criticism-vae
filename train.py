from arguments import prepare_parser
from vae_model.vae import VaeModel
import torch
from dataset_dataloader import ImageDataset
from lagrangian_opt.constraint import ConstraintOptimizer
from objective import Objective
import utils


class Trainer:
    def __init__(self, args, dataset, vae_model, device="cpu"):

        self.device = device
        self.args = args
        self.dataset = dataset
        self.data_loaders = dataset.get_train_validation_loaders()
        self.vae_model = vae_model
        self.objective = Objective(args=args, device=self.device)  # this holds the constraint as well
        self.optimisers = self.get_optimisers()

    def get_optimisers(self):
        vae_optimiser = torch.optim.Adam(self.vae_model.parameters(), lr=self.args.lr)

        if self.args.objective == "MDR-VAE":
            # noinspection PyTypeChecker
            mdr_constraint_optimiser = ConstraintOptimizer(torch.optim.RMSprop,
                                                           self.objective.mdr_constraint.parameters(),
                                                           self.args.mdr_constraint_optim_lr)
            return dict(vae_optimiser=vae_optimiser, mdr_constraint_optimiser=mdr_constraint_optimiser)
        else:
            return dict(vae_optimiser=vae_optimiser)

    def shared_step(self, batch):
        x_in, labels = batch[0], batch[1]

        x_in = x_in.to(self.device)

        q_z_x, z_post, p_z, p_x_z = self.vae_model.forward(x_in)

        loss_dict = self.objective.compute_loss(x_in, q_z_x, z_post, p_z, p_x_z)

        return loss_dict

    def train_step(self, batch):
        for _, o in self.optimisers.items():
            o.zero_grad()

        # Forward
        loss_dict = self.shared_step(batch)

        # Backward
        loss_dict["total_loss"].backward()

        # Step
        for _, o in self.optimisers.items():
            o.step()

        return loss_dict

    def validation_step(self, batch):
        with torch.no_grad():
            return self.shared_step(batch)

    def train(self):
        epoch, step = 0, 0

        for epoch in range(1000):
            for phase in ["train", "valid"]:
                for batch_idx, batch in enumerate(self.data_loaders[phase]):

                    if phase == "train":
                        loss_dict = self.train_step(batch)
                    else:
                        loss_dict = self.validation_step(batch)

                    if self.args.logging and step % self.args.log_every_n_steps == 0:
                        utils.log_step(loss_dict, step, epoch, phase)

                    if self.args.print_stats and step % self.args.print_every_n_steps == 0:
                        utils.print_step(step, epoch, batch_idx, len(self.data_loaders[phase]),
                                         phase, self.args, loss_dict)

                    if phase == "train":
                        step += 1

            epoch += 1

            if epoch == self.args.max_epochs:
                break


def main():
    args = prepare_parser(print_settings=True)

    device_name = utils.determine_device(args)

    dataset = ImageDataset(args=args) if args.image_or_language == "image" else None

    vae_model = VaeModel(args=args, device=device_name)
    vae_model = vae_model.to(device_name)

    if args.logging:
        utils.init_logging(vae_model, args)

    trainer = Trainer(args=args, dataset=dataset, vae_model=vae_model, device=device_name)

    trainer.train()


if __name__ == "__main__":
    main()
