from arguments import prepare_parser
from vae_model.vae import VaeModel
import torch
from dataset_dataloader import ImageDataset
from pytorch_constrained_opt.constraint import ConstraintOptimizer
from objective import Objective
import utils
import torch.optim as optim
from functools import partial


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
        inf_optimiser = self.init_optimiser(self.args.inf_opt, self.vae_model.inf_model.parameters(), self.args.inf_lr,
                                            self.args.inf_l2_weight, self.args.inf_momentum)
        gen_optimiser = self.init_optimiser(self.args.gen_opt, self.vae_model.gen_model.parameters(),
                                            self.args.gen_lr, self.args.gen_l2_weight, self.args.gen_momentum)

        if self.args.objective == "MDR-VAE":
            # noinspection PyTypeChecker
            mdr_constraint_optimiser = ConstraintOptimizer(torch.optim.RMSprop,
                                                           self.objective.mdr_constraint.parameters(),
                                                           self.args.mdr_constraint_optim_lr)
            return dict(inf_optimiser=inf_optimiser, gen_optimiser=gen_optimiser,
                        mdr_constraint_optimiser=mdr_constraint_optimiser)
        else:
            return dict(inf_optimiser=inf_optimiser, gen_optimiser=gen_optimiser)

    @staticmethod
    def init_optimiser(name, parameters, lr, l2_weight, momentum=0.0):
        # Taken from Wilker: https://github.com/probabll/dgm.pt/blob/
        # 95b5b1eb798b87c3d621e7416cc1c423c076c865/probabll/dgm/opt_utils.py#L64
        if name is None or name == "adam":
            cls = optim.Adam
        elif name == "amsgrad":
            cls = partial(optim.Adam, amsgrad=True)
        elif name == "adagrad":
            cls = optim.Adagrad
        elif name == "adadelta":
            cls = optim.Adadelta
        elif name == "rmsprop":
            cls = partial(optim.RMSprop, momentum=momentum)
        elif name == "sgd":
            cls = optim.SGD
        else:
            raise ValueError("Unknown optimizer: %s" % name)
        return cls(params=parameters, lr=lr, weight_decay=l2_weight)

    def shared_step(self, x_in):
        self.vae_model.eval()

        x_in = x_in.to(self.device)

        q_z_x, z_post, p_z, p_x_z = self.vae_model.forward(x_in)

        loss_dict = self.objective.compute_loss(x_in, q_z_x, z_post, p_z, p_x_z)

        return loss_dict

    def train_step(self, x_in):
        self.vae_model.train()

        for _, o in self.optimisers.items():
            o.zero_grad()

        # Forward
        loss_dict = self.shared_step(x_in)

        # Backward
        loss_dict["total_loss"].backward()

        if self.args.max_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters=self.vae_model.parameters(),
                                           max_norm=self.args.max_gradient_norm, norm_type=float("inf"))

        # Step
        for _, o in self.optimisers.items():
            o.step()

        return loss_dict

    def validation_step(self, batch):
        with torch.no_grad():
            return self.shared_step(batch)

    def train(self):
        epoch, step = 0, 0

        best_val_loss, mean_reduced_epoch_stats = 10000.0, None

        for epoch in range(1000):
            for phase in ["train", "valid"]:

                epoch_stats = utils.make_nested_dict()

                for batch_idx, batch in enumerate(self.data_loaders[phase]):
                    # [B, C, W, H]
                    x_in, _ = batch

                    if phase == "train":
                        loss_dict = self.train_step(x_in)
                    else:
                        loss_dict = self.validation_step(x_in)

                    epoch_stats = utils.insert_epoch_stats(epoch_stats, loss_dict)

                    # Log train step batch statistics batch
                    if self.args.logging and step % self.args.log_every_n_steps == 0:
                        utils.log_step(loss_dict, step, epoch, phase)

                    # Print train / validation step batch statistics
                    if self.args.print_stats and step % self.args.print_every_n_steps == 0:
                        utils.print_step(step, epoch, batch_idx, len(self.data_loaders[phase]),
                                         phase, self.args, loss_dict)

                    # Increment step
                    if phase == "train":
                        step += 1

                    # For testing break out early
                    if self.args.short_dev_run and batch_idx == 2:
                        break

                if phase == "valid" and epoch % self.args.eval_ll_every_n_epochs == 0:
                    iw_lls = self.vae_model.estimate_log_likelihood(self.data_loaders["valid"],
                                                                    n_samples=self.args.iw_n_samples,
                                                                    short_dev_run=self.args.short_dev_run)
                    epoch_stats["iw_ll"] = iw_lls

                # W&B Log epoch statistics as <phase>_epoch/<metric>
                if self.args.logging:
                    mean_reduced_epoch_stats = utils.reduce_and_log_epoch_stats(epoch_stats, phase, epoch, step,
                                                                                print_stats=self.args.print_stats)

                if phase == "valid" and mean_reduced_epoch_stats is not None:
                    if mean_reduced_epoch_stats["total_loss"] < best_val_loss:
                        best_val_loss = mean_reduced_epoch_stats["total_loss"]
                        if self.args.checkpointing:
                            utils.make_checkpoint(self.vae_model, self.args, self.optimisers, epoch, step, mean_reduced_epoch_stats)

            # TODO: utils.log_mog(self.vae_model, self.args)

            epoch += 1

            if self.args.short_dev_run and epoch == 2:
                break

            if epoch == self.args.max_epochs:
                break


def main(config=None):
    if config is None:
        args = prepare_parser(print_settings=True)
    else:
        print("USING CONFIG")
        args = config

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
